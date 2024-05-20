import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB

# Load the dataset
df = pd.read_csv('Classification/preprocessed_data/PredictiveM.csv')

# Define target variable
target_variable = "Failure Type"

# Function to train Naive Bayes model
def NaiveBayesModel(df, target_variable):
    X = df.drop(columns=[target_variable])
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

# Function to evaluate classification
def evaluate_classification(model, df, target_variable):
    X = df.drop(columns=[target_variable])
    y = df[target_variable]
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, average='weighted')
    recall = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    return accuracy, precision, recall, f1

# Train Naive Bayes model
model = NaiveBayesModel(df, target_variable=target_variable)

# Evaluate the model
accuracy, precision, recall, f1 = evaluate_classification(model, df, target_variable)

# Print evaluation metrics
print("Accuracy for the Naive Bayes model: {:.2f}".format(accuracy))
print("Precision for the Naive Bayes model: {:.2f}".format(precision))
print("Recall for the Naive Bayes model: {:.2f}".format(recall))
print("F1 Score for the Naive Bayes model: {:.2f}".format(f1))
