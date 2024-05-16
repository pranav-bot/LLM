from utils import perform_eda, prepreocess
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process.kernels import RBF  # Radial Basis Function kernel
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv('./csv_data/heart.csv')
target_variable = 'output'  
kernel=RBF(length_scale=1.0)
learning_rate = 'optimal'
loss = 'log_loss'
penalty = 'elasticnet'
alpha = 0.0001
max_iter = 100
n_neighbors = 5
weights = 'uniform'
algorithm = 'auto'

df = prepreocess(df, perform_eda(df))

X = df.drop(columns=[target_variable])
y = df[target_variable]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def DecisionTree_Classifier(df, target_variable):
  """
  This function trains and evaluates a Decision Tree Classifier.

  Args:
      df (pandas.DataFrame): The preprocessed DataFrame.
      target_variable (str): The name of the target variable (classification label).
      max_depth (int, optional): The maximum depth of the tree. Defaults to 3.
      criterion (str, optional): The function to measure the quality of a split. Defaults to 'gini' (Gini impurity).
      random_state (int, optional): Seed for random number generation. Defaults to None.

  Returns:
      DecisionTreeClassifier: The trained Decision Tree Classifier model.
  """

  model = DecisionTreeClassifier()
  model.fit(X_train, y_train)
  return model

def GPC_Classifier(df, target_variable, kernel= kernel):
  """
  This function trains and evaluates a Gaussian Process Classifier.

  Args:
      df (pandas.DataFrame): The preprocessed DataFrame.
      target_variable (str): The name of the target variable (classification label).
      kernel (object, optional): Kernel function to use. Defaults to RBF(length_scale=1.0).

  Returns:
      GaussianProcessClassifier: The trained GPC model.
  """
  X = df.drop(columns=[target_variable])
  y = df[target_variable]
  model = GaussianProcessClassifier(kernel=kernel)
  model.fit(X_train, y_train)
  return model

def SGD_Classifier(df, target_variable, learning_rate= learning_rate, loss= loss, penalty= penalty, alpha= alpha, max_iter= max_iter, tol=None):
  """
  This function trains and evaluates an SGD classifier.

  Args:
      df (pandas.DataFrame): The preprocessed DataFrame.
      target_variable (str): The name of the target variable (classification label).
      learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.01.
      loss (str, optional): Loss function to use. Defaults to 'log_loss' (logistic loss).
      penalty (str, optional): Regularization method. Defaults to 'l2' (L2 regularization).
      alpha (float, optional): Strength of the regularization parameter. Defaults to 0.0001.
      max_iter (int, optional): Maximum number of iterations. Defaults to 100.
      tol (float, optional): Tolerance for stopping criteria. Defaults to None.

  Returns:
      SGDClassifier: The trained SGD classifier model.
  """
  X = df.drop(columns=[target_variable])
  y = df[target_variable]
  model = SGDClassifier(loss=loss, penalty=penalty, alpha=alpha, max_iter=max_iter, tol=tol, learning_rate=learning_rate)
  model.fit(X_train, y_train)
  return model

def KNN_Classifier(df, target_variable, n_neighbors= n_neighbors, weights= weights, algorithm= algorithm):
  """
  This function trains and evaluates a K-Nearest Neighbors classifier.

  Args:
      df (pandas.DataFrame): The preprocessed DataFrame.
      target_variable (str): The name of the target variable (classification label).
      n_neighbors (int, optional): Number of neighbors to consider for prediction. Defaults to 5.
      weights (str, optional): Weighting scheme for neighbor points. Defaults to 'uniform'.
      algorithm (str, optional): Algorithm for neighbor search. Defaults to 'auto'.

  Returns:
      KNeighborsClassifier: The trained KNN classifier model.
  """
  X = df.drop(columns=[target_variable])
  y = df[target_variable]
  model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
  model.fit(X_train, y_train)
  return model

def GaussianNB_Classifier(df, target_variable):
  """
  This function trains and evaluates a Gaussian Naive Bayes classifier.

  Args:
      df (pandas.DataFrame): The preprocessed DataFrame.
      target_variable (str): The name of the target variable (classification label).

  Returns:
      GaussianNB: The trained Gaussian Naive Bayes classifier model.
  """
  X = df.drop(columns=[target_variable])
  y = df[target_variable]
  model = GaussianNB()
  model.fit(X_train, y_train)
  return model

def evaluate_classification(model, df):
  """
  This function evaluates the performance of a classification model.

  Args:
      model (object): The trained classification model.
      df (pandas.DataFrame): The DataFrame to use for evaluation.

  Returns:
      dict: A dictionary containing the evaluation metrics (accuracy, precision, recall, F1-score).
  """
  X = df.drop(columns=[target_variable])
  y = df[target_variable]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  y_pred = model.predict(X_test)
  accuracy = accuracy_score(y_test, y_pred)
  precision = precision_score(y_test, y_pred, average='weighted')  # Consider using other averaging options
  recall = recall_score(y_test, y_pred, average='weighted')  # Consider using other averaging options
  f1 = f1_score(y_test, y_pred, average='weighted')  # Consider using other averaging options
  return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}

models = ["DecisionTree_Classifier", 'GPC_Classifier', 'SGD_Classifier', 'KNN_Classifier', 'GaussianNB_Classifier']

for model_name in models:

  model = eval(model_name)(df, target_variable)

  evaluation_results = evaluate_classification(model, df.copy())

  print(f"Accuracy for the {model_name}:", evaluation_results['accuracy'])
  print("Precision for the Decision Tree Classifier:", evaluation_results['precision'])
  print("Recall for the Decision Tree Classifier:", evaluation_results['recall'])
  print("F1-score for the Decision Tree Classifier:", evaluation_results['f1'])
