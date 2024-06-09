# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="meta-llama/Llama-2-7b-chat-hf")

pipeline (' I have a dream that one day this nation will rise up and live out the true meaning of its creed.')