from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd

data = pd.read_csv("Drug.csv")

features = data.columns.tolist()
print(features)

# Prepare the input as before
chat = [
    {"role": "system", "content": "You are a machine lerning expert"},
    {"role": "user", "content": f"This dataset contains information about [dataset topic]. Analyze the feature '{features}' and describe its relevance to understanding [target variable] or achieving the overall goal."}
]

# 1: Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("davidkim205/Rhea-72b-v0.5")
model = AutoModelForCausalLM.from_pretrained("davidkim205/Rhea-72b-v0.5")

# 2: Apply the chat template
formatted_chat = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
print("Formatted chat:\n", formatted_chat)

# 3: Tokenize the chat (This can be combined with the previous step using tokenize=True)
inputs = tokenizer(formatted_chat, return_tensors="pt", add_special_tokens=False)
# Move the tokenized inputs to the same device the model is on (GPU/CPU)
inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
print("Tokenized inputs:\n", inputs)

# 4: Generate text from the model
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.)
print("Generated tokens:\n", outputs)

# 5: Decode the output back to a string
decoded_output = tokenizer.decode(outputs[0][inputs['input_ids'].size(1):], skip_special_tokens=True)
print("Decoded output:\n", decoded_output)