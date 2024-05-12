#!/usr/bin/env python3
#

import pandas as pd
from utils import prepreocess, perform_eda


df = pd.read_csv('./csv_data/waterquality.csv')

df = prepreocess(df, perform_eda(df))
df.to_csv('./csv_data/waterquality_preprocessed.csv')


print(df)
