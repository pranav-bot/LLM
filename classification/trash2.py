import pandas as pd
from utils import perform_eda, prepreocess
import glob
import google.generativeai as genai
import time
import json
import os
from dotenv import load_dotenv


current_dir = "classification/classification_data"
files = os.listdir(current_dir)

print(files)

with open('Classification/file_names.txt', 'w') as f:
   for item in sorted(files):
      f.write(f"{item}\n")
