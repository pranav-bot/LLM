from utils import perform_eda, prepreocess
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv('./csv_data/titanic.csv')
target_variable = "Survived"  # Assuming this is the classification label
n_neighbors = 5
weights = 'uniform'
algorithm = 'auto'

df = prepreocess(df, perform_eda(df))

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
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
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

# Train the KNN model
model = KNN_Classifier(df, target_variable)

# Evaluate the KNN model
evaluation_results = evaluate_classification(model, df.copy())

print("Accuracy for the KNN Classifier:", evaluation_results['accuracy'])
print("Precision for the KNN Classifier:", evaluation_results['precision'])
print("Recall for the KNN Classifier:", evaluation_results['recall'])
print("F1-score for the KNN Classifier:", evaluation_results['f1'])
