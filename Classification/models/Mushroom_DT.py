import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier

# Load the dataset
df = pd.read_csv('Classification/preprocessed_data/Mushroom.csv')

# Define target variable
target_variable = "class"

# Function to train Decision Tree model
def DecisionTreeModel(df, target_variable):
    X = df.drop(columns=[target_variable])
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model

# Function to evaluate classification
def evaluate_classification(model, df, target_variable):
    X = df.drop(columns=[target_variable])
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, precision, recall, f1

# Train Decision Tree model
model = DecisionTreeModel(df, target_variable=target_variable)

# Evaluate the model
accuracy, precision, recall, f1 = evaluate_classification(model, df, target_variable)

# Print evaluation metrics
print("Accuracy for the Decision Tree model: {:.2f}".format(accuracy))
print("Precision for the Decision Tree model: {:.2f}".format(precision))
print("Recall for the Decision Tree model: {:.2f}".format(recall))
print("F1 Score for the Decision Tree model: {:.2f}".format(f1))
