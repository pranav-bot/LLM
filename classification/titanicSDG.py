from utils import perform_eda, prepreocess
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv('./csv_data/titanic.csv')
target_variable = "Survived"  # Assuming this is the classification label
learning_rate = 'optimal'
loss = 'log_loss'
penalty = 'elasticnet'
alpha = 0.0001
max_iter = 100

df = prepreocess(df, perform_eda(df))

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
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  model = SGDClassifier(loss=loss, penalty=penalty, alpha=alpha, max_iter=max_iter, tol=tol, learning_rate=learning_rate)
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

# Train the SGD classifier
model = SGD_Classifier(df, target_variable)

# Evaluate the SGD classifier
evaluation_results = evaluate_classification(model, df.copy())

print("Accuracy for the SGD Classifier:", evaluation_results['accuracy'])
print("Precision for the SGD Classifier:", evaluation_results['precision'])
print("Recall for the SGD Classifier:", evaluation_results['recall'])
print("F1-score for the SGD Classifier:", evaluation_results['f1'])
