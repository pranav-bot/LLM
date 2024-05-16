import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

model_names = ['Linear', 'Ridge', 'Lasso', 'Bayesian', 'SGD', 'KernelRidge', 'SVM', 'KNeighbors', 'GaussianProcessRegressor', 'PLS', 'DecisionTree', 'MLP', 'RandomForest']

datasets = ["winequality-red", "Student_Performance", "kc_house_data",]

target_variables = {
    "winequality-red": ["quality"],
    "Student_Performance": ["Performance Index"],
    "kc_house_data": ["price"],


}

columns_to_drop={
    "winequality-red": "",
    "Student_Performance": "",
    "kc_house_data": "'id', 'date',",
}

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
    'PLS': 'from sklearn.cross_decomposition import PLSRegression',
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
    'PLS': 'PLSRegression',
    'DecisionTree': 'DecisionTreeRegressor',
    'MLP': 'MLPRegressor',
    'RandomForest': 'RandomForestRegressor'
}

prompt_code_pairs = []

for data in datasets:
    df = pd.read_csv('./csv_data2/' + data + '.csv')
    columns = list(df.columns)

    for target_variable in target_variables[data]:
        for model in model_names:
            prompt = "Write python functions to make a machine learning {} regression model based on a dataset named {} having columns {} with the target variable being {} having the regression evaluation metrics mse,mae,r2,rmse"
            print(prompt.format(model, data, columns, target_variable))

            code = """
import pandas as pd
{}
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_squared_error


df = pd.read_csv('./csv_data/{}_preprocessed.csv')
target_variable = '{}'

def {}RegressionModel(df, target_variable):
    X = df.drop(columns=[{}target_variable])
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = {}()
    model.fit(X_train, y_train)
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

print("MSE for the Regression model: "+mse)
print("MAE for the Regression model: "+mae)
print("R2 for the Regression model: "+r2)
print("RMSE for the Regression model: "+rmse)
"""
            to_import = imports[model]
            model_name = model
            _columns_to_drop = columns_to_drop[data]
            model_class = model_classes[model]
            print(code.format(to_import, data,target_variable,model,_columns_to_drop, model_class, _columns_to_drop,model))
            prompt_code_pairs.append((prompt.format(model, data, columns, target_variable),code.format(to_import, data,target_variable,model,_columns_to_drop, model_class, _columns_to_drop,model)))

df_pairs = pd.DataFrame(prompt_code_pairs, columns=['Prompt', 'Code'])

df_pairs.to_csv('regression_data.csv', index=False)
