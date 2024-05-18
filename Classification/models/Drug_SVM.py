import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
df = pd.read_csv('Classification/preprocessed_data/Drug.csv')

# Define target variable
target_variable = "Drug"


# Function to train SVM model
def SVMModel(df, target_variable, kernel='rbf', C=1.0, gamma='scale'):
    X = df.drop(columns=[target_variable])
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC(kernel=kernel, C=C, gamma=gamma)
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

# Train SVM model
model = SVMModel(df, target_variable=target_variable)

# Evaluate the model
accuracy, precision, recall, f1 = evaluate_classification(model, df)

# Print evaluation metrics
print("Accuracy for the SVM model: {:.2f}".format(accuracy))
print("Precision for the SVM model: {:.2f}".format(precision))
print("Recall for the SVM model: {:.2f}".format(recall))
print("F1 Score for the SVM model: {:.2f}".format(f1))
