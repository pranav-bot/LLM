#!/usr/bin/env python3

import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error


df = pd.read_csv('./csv_data/waterquality_preprocessed.csv')
target_variable = "AirTemp (C)"

def SVMRegressionModel(df, target_variable, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=1e-3, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=-1):
    X = df.drop(columns=['Date', target_variable])
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=42)
    model = SVR(kernel=kernel,degree=degree,gamma=gamma,coef0=coef0, tol=tol, C=C, epsilon=epsilon, shrinking=shrinking, cache_size=cache_size, verbose=verbose,max_iter=max_iter)
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

model = SVMRegressionModel(df, target_variable=target_variable)

mse, r2, mae, rmse = evaluate_regression(model, df)

print("MSE for the Linear Regression model:"+str(mse))
print("MAE for the Linear Regression model:"+str(mae))
print("R2 for the Linear Regression model:"+str(r2))
print("RMSE for the Linear Regression model:"+str(rmse))
