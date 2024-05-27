import ipywidgets as widgets
import openvino as ov
import time
import numpy as np
import gradio as gr
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

core = ov.Core()

device = widgets.Dropdown(
    options=core.available_devices + ["AUTO"],
    value="CPU",
    description="Device:",
    disabled=False,
)

def autoregressive_sampling_with_pkv(input, model, N=30):
    input_ids, attention_mask = input.input_ids, input.attention_mask
    seq_len = input_ids.shape[-1]
    position_ids = np.arange(0, seq_len, dtype=np.int64).reshape([-1, seq_len])

    # in all subsequent inferences we feed tokens one by one,
    # but for the first one we feed the whole encoded prompt
    request = model.create_infer_request()
    request.infer((input_ids, attention_mask, position_ids, np.array([0])))
    next_token = np.argmax(request.results["logits"][:, -1]).reshape([1])

    all_tokens = []
    all_tokens.extend(input_ids[0])
    all_tokens.append(next_token[0])

    while seq_len < N:
        input_ids = next_token.reshape([1, 1])
        attention_mask = np.concatenate((attention_mask, np.array([1]).reshape([1, 1])), axis=1)
        position_ids = np.array([attention_mask.shape[1]]).reshape([1, 1])

        request.infer((input_ids, attention_mask, position_ids, np.array([0])))
        next_token = np.argmax(request.results["logits"][:, -1])
        all_tokens.append(next_token)
        seq_len += 1

    return all_tokens

def update_state(request, seq_len):
    for state in request.query_state():
        old_seq_len = state.state.shape[2]
        if seq_len >= old_seq_len:
            continue
        # After the inference request, key/values have shape [BATCH_SIZE, seq_len + K, vocab_size].
        # Increment the sequence length by the number of matched tokens, and
        # trim the KV cache to match the new sequence length.
        state.state = ov.Tensor(state.state.data[:, :, :seq_len])


def speculative_sampling_with_pkv(input, draft_model, main_model, K, N=30, **kwargs):
    input_ids, attention_mask = input.input_ids, input.attention_mask
    # seq_len number of key/values or number of already processed input tokens
    seq_len = input_ids.shape[-1]
    position_ids = np.arange(0, seq_len, dtype=np.int64).reshape([-1, seq_len])

    draft_request = draft_model.create_infer_request()
    draft_request.infer((input_ids, attention_mask, position_ids, np.array([0])))

    main_request = main_model.create_infer_request()
    main_request.infer((input_ids, attention_mask, position_ids, np.array([0])))
    first_token = np.argmax(main_request.results["logits"][:, -1]).reshape([1])

    all_tokens = []
    all_tokens.extend(input_ids[0])
    all_tokens.append(first_token[0])

    accum_draft_tokens = []
    while seq_len < N:
        next_token = first_token
        for i in range(K):
            input_ids = next_token.reshape([1, 1])
            attention_mask = np.concatenate((attention_mask, np.array([1]).reshape([1, 1])), axis=1)
            position_ids = np.array([attention_mask.shape[1]]).reshape([1, 1])

            draft_request.infer((input_ids, attention_mask, position_ids, np.array([0])))
            next_token = np.argmax(draft_request.results["logits"][:, -1])
            accum_draft_tokens.append(next_token)

        # main model will give also K out tokens
        # feed the same first token to the main model and do not give the last token generated by the draft
        input_ids = np.concatenate((first_token.reshape([1]), accum_draft_tokens[:-1])).reshape([1, -1])
        attention_mask = np.ones((1, seq_len + K))
        position_ids = np.arange(seq_len, seq_len + K, dtype=np.int64).reshape([1, -1])

        main_request.infer((input_ids, attention_mask, position_ids, np.array([0])))
        next_tokens = np.argmax(main_request.results["logits"], axis=-1)[0]

        # if disagrees from the very beggining then context will be expanded only for one element
        # all elements match then context will be expanded to K elements
        for disagree_idx, (t1, t2) in enumerate(zip(accum_draft_tokens, next_tokens)):
            if t1 != t2:
                break

        first_token = next_tokens[disagree_idx]
        all_tokens.extend(next_tokens[: disagree_idx + 1])
        seq_len += disagree_idx + 1

        # cut key/values depending on the position where disagreement starts
        update_state(draft_request, seq_len)
        update_state(main_request, seq_len)

        attention_mask = np.ones((1, seq_len))
        accum_draft_tokens = []
    all_tokens.extend(accum_draft_tokens)
    return all_tokens

main_model_id = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
main_model_path = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
draft_model_id = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
draft_model_path = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

main_tokenizer = AutoTokenizer.from_pretrained(main_model_id)
draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_id)

# In order for speculative sampling to work, both main and draft tokenizers should be the same.
token_test_txt = "text to ensure tokenizers work the same, as of 2024"
tokens_1 = draft_tokenizer(token_test_txt, return_tensors="pt").input_ids
tokens_2 = main_tokenizer(token_test_txt, return_tensors="pt").input_ids

assert all((tokens_1 - tokens_2)[0] == 0)

def main(
    prompt: str,
    n_tokens_to_generate: int = 75,
    K: int = 5,
    seed: int = 5555,
):
    # seed numpy rng
    np.random.seed(seed)
    tokenized = main_tokenizer(prompt, return_tensors="pt")

    def run_autoregressive_sampling_fn(decode_fn, tokenized, **kwargs):
        start = time.perf_counter()
        output_ids = decode_fn(tokenized, **kwargs)
        text = main_tokenizer.decode(output_ids, skip_special_tokens=True)
        elapsed_time = time.perf_counter() - start
        return text, elapsed_time

    def run_speculative_sampling_fn(decode_fn, input_ids, **kwargs):
        start = time.perf_counter()
        output_ids = decode_fn(input_ids, **kwargs)
        text = main_tokenizer.decode(output_ids, skip_special_tokens=True)
        elapsed_time = time.perf_counter() - start
        return text, elapsed_time

    autoregressive_text, autoregressive_time = run_autoregressive_sampling_fn(
        autoregressive_sampling_with_pkv,
        tokenized,
        model=main_model,
        N=n_tokens_to_generate,
    )

    speculative_text, speculative_time = run_speculative_sampling_fn(
        speculative_sampling_with_pkv,
        tokenized,
        main_model=main_model,
        draft_model=draft_model,
        N=n_tokens_to_generate,
        K=K,
    )

    # Format results for output in gradio
    out = "\n" + "Autoregressive Decode" + "\n" + "---------------------" + "\n"
    out = out + f"Time = {autoregressive_time:.2f}s" + "\n" + f"Text = {autoregressive_text}" + "\n"
    out = out + "\n" + "Speculative Decode" + "\n" + "------------------" + "\n"
    out = out + f"Time = {speculative_time:.2f}s" + "\n" + f"Text = {speculative_text}"
    return out

res = main("Alan Turing was a", n_tokens_to_generate=100)
print(res)