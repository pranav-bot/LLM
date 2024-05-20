import pandas as pd
from transformers import GPT2TokenizerFast

# Load your dataset
data = pd.read_csv("Drug.csv")

# Select features (assuming you want to analyze all)
features = data.columns.tolist()

# Initialize LLM pipeline (replace 'text-davinci-003' with your preferred model)
tokenizer = GPT2TokenizerFast.from_pretrained('Xenova/text-davinci-003')

def analyze_feature(feature_name):
  # Prepare prompt (replace with your specific prompt template)
  prompt = f"This dataset contains information about [dataset topic]. Analyze the feature '{feature_name}' and describe its relevance to understanding [target variable] or achieving the overall goal."

  # Send prompt to LLM and get response
  response = tokenizer(prompt, max_length=100, truncation=True)[0]["generated_text"]
  print(f"Feature: {feature_name}\nLLM Analysis: {response}\n")

# Analyze each feature
for feature in features:
  analyze_feature(feature)

# Additional steps: Refine prompts based on LLM responses, make decisions about dropping features

