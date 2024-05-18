import pandas as pd
from utils import perform_eda, prepreocess
import glob
import google.generativeai as genai
import time
import json
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('API_KEy')


genai.configure(api_key=API_KEY)

current_dir = "regression/csv_data"
files = os.listdir(current_dir)

file_list = []

target_variables = {}
columns_to_drop = {}
counter = 1

for file in sorted(files):
  try:
    if counter%5==0:
      time.sleep(60)  
    df = pd.read_csv("regression/csv_data/"+ file)
    df = prepreocess(df, perform_eda(df))
    file_list.append(file)
    columns = list(df.columns)
    prompt = "Given a list of columns {} identify the columns that could potentially be used as target variables for regression and return only those column names. For example, if a column contains timestamps it's unlikely to be a target variable return the columns in a python list format [] dont write code only according to you what columns can be used"
    prompt = prompt.format(columns)
    prompt2 = "Given a list of columns {} identify the columns that could potentially be dropped because they likely wouldn't be useful for analysis and return only those column names in a python list format []. For example, columns with constant values or timestamps are often dropped.Drop only if you think drooping that columns is neccessary. Don't write code or explanation, just return the list of columns, if no columns are to be dropped return only [] "
    prompt2 = prompt2.format(columns)
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    response2 = model.generate_content(prompt2)
    try:
       text_content = response.candidates[0].content.parts[0].text
    except IndexError:
       text_content = "[]" 
    try:
        text_content2 = response2.candidates[0].content.parts[0].text
    except IndexError:
        text_content2 = "[]"
    target_variables[file] = text_content
    columns_to_drop[file] = text_content2
    counter += 1
  except Exception as e:
    print(e)
    print(file)

def process(data_dict):
  for key, value_string in data_dict.items():
     value_string = value_string.strip("[]")
     value_list = value_string.split(",")
     value_list = [item.strip() for item in value_list]
     data_dict[key] = value_list

  for key, value_list in data_dict.items():
     data_dict[key] = [item.strip("'").strip() for item in value_list]  
 
  return data_dict

target_variables = process(target_variables)
columns_to_drop = process(columns_to_drop)

print(columns_to_drop)
print(target_variables)

with open("regression/target_variables.json", "w") as outfile: 
    json.dump(target_variables, outfile)

with open("regression/columns_to_drop.json", "w") as outfile: 
    json.dump(columns_to_drop, outfile)

with open('regression/file_names.txt', 'w') as f:
   for item in file_list:
      f.write(f"{item}\n")