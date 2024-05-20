import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
df = pd.read_csv('Classification/preprocessed_data/Obesity.csv')

# Define target variable
target_variable = "Label"

# Function to train logistic regression model
def LogisticRegressionModel(df, target_variable, fit_intercept=True, solver='liblinear'):
    X = df.drop(columns=[target_variable])
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(fit_intercept=fit_intercept, solver=solver)
    model.fit(X_train, y_train)
    return model

# Function to evaluate classification
def evaluate_classification(model, df):
    X = df.drop(columns=[target_variable])
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1

# Train logistic regression model
model = LogisticRegressionModel(df, target_variable=target_variable)

# Evaluate the model
accuracy, precision, recall, f1 = evaluate_classification(model, df)

# Print evaluation metrics
print("Accuracy for the Logistic Regression model: {:.2f}".format(accuracy))
print("Precision for the Logistic Regression model: {:.2f}".format(precision))
print("Recall for the Logistic Regression model: {:.2f}".format(recall))
print("F1 Score for the Logistic Regression model: {:.2f}".format(f1))
