from utils import perform_eda, prepreocess
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv('./csv_data/titanic.csv')
target_variable = "Survived"  # Assuming this is the classification label

df = prepreocess(df, perform_eda(df))

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
  X = df.drop(columns=[target_variable])
  y = df[target_variable]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  model = DecisionTreeClassifier()
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

# Train the Decision Tree Classifier
model = DecisionTree_Classifier(df, target_variable)

# Evaluate the Decision Tree Classifier
evaluation_results = evaluate_classification(model, df.copy())

print("Accuracy for the Decision Tree Classifier:", evaluation_results['accuracy'])
print("Precision for the Decision Tree Classifier:", evaluation_results['precision'])
print("Recall for the Decision Tree Classifier:", evaluation_results['recall'])
print("F1-score for the Decision Tree Classifier:", evaluation_results['f1'])
