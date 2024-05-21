import pandas as pd
import json

datasets = []

with open('regression/file_names.txt', 'r') as f:
    lines = f.readlines()
    datasets = [line.strip() for line in lines]


print(datasets)

model_names = ['Linear', 'Ridge', 'Lasso', 'Bayesian', 'SGD', 'KernelRidge', 'SVM', 'KNeighbors', 'GaussianProcessRegressor', 'DecisionTree', 'MLP', 'RandomForest']

target_variables = {}

with open("regression/target_variables.json", 'r') as f:
  target_variables = json.load(f)

print(target_variables)

columns_to_drop={}
with open("regression/columns_to_drop.json", 'r') as f:
  columns_to_drop = json.load(f)

print(columns_to_drop)

imports = {
    'Linear':'from sklearn.linear_model import LinearRegression',
    'Ridge': 'from sklearn.linear_model import Ridge',
    'Lasso': 'from sklearn.linear_model import Lasso',
    'Bayesian': 'from sklearn.linear_model import BayesianRidge',
    'SGD': 'from sklearn.linear_model import SGDRegressor',
    'KernelRidge': 'from sklearn.kernel_ridge import KernelRidge',
    'SVM': 'from sklearn.svm.SVR import SVR',
    'KNeighbors': 'from sklearn.neighbors import KNeighborsRegressor',
    'GaussianProcessRegressor': 'from sklearn.gaussian_process import GaussianProcessRegressor',
    'DecisionTree': 'from sklearn.tree import DecisionTreeRegressor',
    'MLP': 'from sklearn.neural_network import MLPRegressor',
    'RandomForest': 'from sklearn.ensemble import RandomForestRegressor'
}

model_classes = {
    'Linear':'LinearRegression',
    'Ridge': 'Ridge',
    'Lasso': 'Lasso',
    'Bayesian': 'BayesianRidge',
    'SGD': 'SGDRegressor',
    'KernelRidge': 'KernelRidge',
    'SVM': 'SVR',
    'KNeighbors': 'KNeighborsRegressor',
    'GaussianProcessRegressor': 'GaussianProcessRegressor',
    'DecisionTree': 'DecisionTreeRegressor',
    'MLP': 'MLPRegressor',
    'RandomForest': 'RandomForestRegressor'
}

hyperparameters = {
    'Linear': "param_grid = {'fit_intercept': [True, False]}",
    'Ridge': "param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10]}",
    'Lasso': "param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10]}",
    'Bayesian': "param_grid = {'lambda_1': [1e-6, 1e-4, 1e-2],'lambda_2': [1e-6, 1e-4, 1e-2],'tol': [1e-3, 1e-4, 1e-5],}",
    'SGD': "param_grid = {'eta': ['constant', 'invscaling'], 'alpha': [0.0001, 0.001, 0.01],'loss': ['squared_loss', 'huber_loss']}",
    'KernelRidge': "param_grid = {'alpha': [0.001, 0.01, 0.1, 1], 'gamma': [0.1, 1, 10], 'kernel': ['rbf', 'linear', 'poly']}",
    'SVM': "param_grid = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear', 'poly'], 'gamma': [0.1, 1, 10], 'epsilon': [0.001, 0.01, 0.1],}",
    'KNeighbors': "param_grid = {'n_neighbors': [5, 10, 15], 'weights': ['uniform', 'distance'], 'metric': ['minkowski', 'euclidean', 'manhattan']}",
    'GaussianProcessRegressor': "param_grid = {'kernel': ['rbf', 'matern', 'exponential'], 'n_restarts_optimizer': [0, 2, 5], }",
    'DecisionTree': "param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': [3, 5, 8], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}",
    'MLP': "param_grid = {'hidden_layer_sizes': [(100,), (50, 50), (100, 50)], 'activation': ['relu', 'tanh', 'logistic'], 'alpha': [0.0001, 0.001, 0.01], 'solver': ['adam', 'lbfgs']}",
    'RandomForest': "param_grid = {'n_estimators': [100, 200, 500], 'max_depth': [3, 5, 8], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}"
}

prompt_code_pairs = []

for data in datasets:
    df = pd.read_csv('regression/regression_data/' + data)
    columns = list(df.columns)

    for target_variable in target_variables[data]:
        for model in model_names:
            prompt = "Write python functions to make a machine learning {} regression model based on a dataset named {} having columns {} with the target variable being {} having the regression evaluation metrics mse,mae,r2,rmse"
            print(prompt.format(model, data, columns, target_variable))

            code = """
import pandas as pd
{}
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_squared_error


df = pd.read_csv('./csv_data/{}_preprocessed.csv')
target_variable = '{}'

def {}RegressionModel(df, target_variable):
    X = df.drop(columns=[{}target_variable])
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = {}()
    {}
    gs = GridSearchCV(model, param_grid, cv=5)
    gs.fit(X_train, y_train)
    model = gs.best_estimator_
    return model

def evaluate_regression(model, df):
    X = df.drop(columns=[{}target_variable])
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)**0.5
    return mse, r2, mae, rmse

model = {}RegressionModel(df, target_variable=target_variable)

mse, r2, mae, rmse = evaluate_regression(model, df)

print("MSE for the Regression model: "+str(mse))
print("MAE for the Regression model: "+str(mae))
print("R2 for the Regression model: "+str(r2))
print("RMSE for the Regression model: "+str(rmse))
"""
            to_import = imports[model]
            model_name = model
            _columns_to_drop = ""
            for item in columns_to_drop[data]:
               if item!="":
                  _columns_to_drop = _columns_to_drop+"'"+item+"'"+","
            model_class = model_classes[model]
            param_grid = hyperparameters[model]
            print(code.format(to_import, data,target_variable,model,_columns_to_drop, model_class,param_grid, _columns_to_drop,model))
            prompt_code_pairs.append((prompt.format(model, data, columns, target_variable),code.format(to_import, data,target_variable,model,_columns_to_drop, model_class,param_grid, _columns_to_drop,model)))

df_pairs = pd.DataFrame(prompt_code_pairs, columns=['Prompt', 'Code'])

df_pairs.to_csv('regression_data.csv', index=False)
