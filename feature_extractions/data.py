import pandas as pd

data = pd.read_csv("Drug.csv")

# Select features (assuming you want to analyze all)
features = data.columns.tolist()
print(features)