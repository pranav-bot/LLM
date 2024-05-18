from utils import prepreocess, perform_eda
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_squared_error

df = pd.read_csv('regression/csv_data2/kc_house_data.csv')
df = prepreocess(df, perform_eda(df))
target_variable = 'price'

def LinearRegressionModel(df, target_variable):
    X = df.drop(columns=['id', 'date',target_variable])
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    param_grid = {'fit_intercept': [True, False]}
    gs = GridSearchCV(model, param_grid, cv=5)
    gs.fit(X_train, y_train)
    model = gs.best_estimator_
    return model

def evaluate_regression(model, df):
    X = df.drop(columns=['id', 'date',target_variable])
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)**0.5
    return mse, r2, mae, rmse

model = LinearRegressionModel(df, target_variable=target_variable)

mse, r2, mae, rmse = evaluate_regression(model, df)

print("MSE for the Regression model: "+str(mse))
print("MAE for the Regression model: "+str(mae))
print("R2 for the Regression model: "+str(r2))
print("RMSE for the Regression model: "+str(rmse))