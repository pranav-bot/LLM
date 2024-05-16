#!/usr/bin/env python3

from utils import perform_eda, prepreocess
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error


df = pd.read_csv('./csv_data/waterquality_preprocessed.csv')
target_variable = "AirTemp (C)"

def LinearRegressionModel(df, target_variable, fit_intercept=True, copy_X = True, n_jobs=None, positive=False):
    X = df.drop(columns=['Date', target_variable])
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42)
    model = LinearRegression(fit_intercept=fit_intercept, copy_X=copy_X, n_jobs=n_jobs, positive=positive)
    model.fit(X_train, y_train)
    return model

def evaluate_regression(model, df):
    X = df.drop(columns=['Date', target_variable])
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    return mse, r2, mae, rmse

model = LinearRegressionModel(df, target_variable=target_variable)

mse, r2, mae, rmse = evaluate_regression(model, df)

print("MSE for the Linear Regression model:"+str(mse))
print("MAE for the Linear Regression model:"+str(mae))
print("R2 for the Linear Regression model:"+str(r2))
print("RMSE for the Linear Regression model:"+str(rmse))
