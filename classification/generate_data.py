import pandas as pd
import json

datasets = []

with open('Classification/file_names.txt', 'r') as f:
    lines = f.readlines()
    datasets = [line.strip() for line in lines]

print(datasets)

model_names = ['Logistic', 'KNeighbors', 'DecisionTree', 'SVM', 'RandomForest', 'MLP', 'SGD', 'GPC', 'GNB', 'MNB', 'CNB', 'BNB', 'AdaBoost', 'GradientBoosting']

target_variables = {}

with open("classification/target_variables.json", 'r') as f:
    target_variables = json.load(f)

print(target_variables)

columns_to_drop = {}
with open("classification/columns_to_drop.json", 'r') as f:
    columns_to_drop = json.load(f)

print(columns_to_drop)

imports = {
    'Logistic': 'from sklearn.linear_model import LogisticRegression',
    'KNeighbors': 'from sklearn.neighbors import KNeighborsClassifier',
    'DecisionTree': 'from sklearn.tree import DecisionTreeClassifier',
    'SVM': 'from sklearn.svm import SVC',
    'RandomForest': 'from sklearn.ensemble import RandomForestClassifier',
    'MLP': 'from sklearn.neural_network import MLPClassifier',
    'SGD': 'from sklearn.linear_model import SGDClassifier',
    'GPC': 'from sklearn.gaussian_process import GaussianProcessClassifier',
    'GNB': 'from sklearn.naive_bayes import GaussianNB',
    'MNB': 'from sklearn.naive_bayes import MultinomialNB',
    'CNB': 'from sklearn.naive_bayes import ComplementNB',
    'BNB': 'from sklearn.naive_bayes import BernoulliNB',
    'AdaBoost': 'from sklearn.ensemble import AdaBoostClassifier',
    'GradientBoosting': 'from sklearn.ensemble import GradientBoostingClassifier'
}

model_classes = {
    'Logistic': 'LogisticRegression',
    'KNeighbors': 'KNeighborsClassifier',
    'DecisionTree': 'DecisionTreeClassifier',
    'SVM': 'SVC',
    'RandomForest': 'RandomForestClassifier',
    'MLP': 'MLPClassifier',
    'SGD': 'SGDClassifier',
    'GPC': 'GaussianProcessClassifier',
    'GNB': 'GaussianNB',
    'MNB': 'MultinomialNB',
    'CNB': 'ComplementNB',
    'BNB': 'BernoulliNB',
    'AdaBoost': 'AdaBoostClassifier',
    'GradientBoosting': 'GradientBoostingClassifier'
}

hyperparameters = {
    'Logistic': "param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}",
    'KNeighbors': "param_grid = {'n_neighbors': [5, 10, 15], 'weights': ['uniform', 'distance'], 'metric': ['minkowski', 'euclidean', 'manhattan']}",
    'DecisionTree': "param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': [3, 5, 8], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}",
    'SVM': "param_grid = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear', 'poly'], 'gamma': [0.1, 1, 10]}",
    'RandomForest': "param_grid = {'n_estimators': [100, 200, 500], 'max_depth': [3, 5, 8], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}",
    'MLP': "param_grid = {'hidden_layer_sizes': [(100,), (50, 50), (100, 50)], 'activation': ['relu', 'tanh', 'logistic'], 'alpha': [0.0001, 0.001, 0.01], 'solver': ['adam', 'lbfgs']}",
    'SGD': "param_grid = {'alpha': [0.0001, 0.001, 0.01, 0.1], 'loss': ['hinge', 'log'], 'penalty': ['l1', 'l2'], 'C': [0.001, 0.01, 0.1, 1, 10], 'max_iter': [100, 200, 500], 'tol': [1e-3, 1e-4]}",
    'GPC': "param_grid = {'kernel': ['rbf', 'linear', 'matern'], 'n_restarts_optimizer': [0, 10, 20]}",
    'GNB': "param_grid = {}",
    'MNB': "param_grid = {'alpha': [0.0001, 0.001, 0.1, 1, 10], 'fit_prior': [True, False]}",
    'CNB': "param_grid = {'alpha': [0.0001, 0.001, 0.1, 1, 10], 'norm': [True, False]}",
    'BNB': "param_grid = {'alpha': [0.0001, 0.001, 0.1, 1, 10], 'fit_prior': [True, False]}",
    'AdaBoost': "param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.1, 1.0], 'base_estimator__max_depth': [2, 3, 4]}",
    'GradientBoosting': "param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.1, 1.0], 'max_depth': [2, 3, 4], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 3, 5]}"
}

prompt_code_pairs = []

for data in datasets:
    df = pd.read_csv('classification/classification_data/' + data)
    columns = list(df.columns)

    for target_variable in target_variables[data]:
        for model in model_names:
            prompt = "Write python functions to make a machine learning {} classification model based on a dataset named {} having columns {} with the target variable being {} having the classification evaluation metrics accuracy, precision, recall, f1 score"
            print(prompt.format(model, data, columns, target_variable))

            code = f"""
import pandas as pd
{imports[model]}
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv('./classification/classification_data/{data}')
target_variable = '{target_variable}'

def {model}ClassificationModel(df, target_variable):
    X = df.drop(columns={columns_to_drop.get(data, [])} + [target_variable])
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = {model_classes[model]}()
    {hyperparameters[model]}
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    y_pred = best_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    metrics = {{
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }}
    
    return best_model, metrics
            """

            # Print the generated code
            print(code)
            prompt_code_pairs.append((prompt.format(model, data, columns, target_variable), code))


df_pairs = pd.DataFrame(prompt_code_pairs, columns=['Prompt', 'Code'])


df_pairs.to_csv('classification_data.csv', index=False)
