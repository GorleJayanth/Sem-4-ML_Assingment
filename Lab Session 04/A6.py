import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

def load_dataset(file_path):
    """Loads dataset from an Excel file and checks if it exists."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' was not found in the directory.")
    
    df = pd.read_excel(file_path)
    
    if "Label" not in df.columns:
        raise ValueError("Error: 'Label' column not found in the dataset. Available columns are:", df.columns)
    
    df = df[df["Label"].isin([0, 1])]  # Filter for labels 0 and 1

    if df.empty:
        raise ValueError("Error: No data available after filtering for labels 0 and 1.")
    
    return df

def preprocess_data(df):
    """Extracts features and labels from the dataset."""
    X = df.drop(columns=["Label"]).values
    y = df["Label"].values

    if X.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("Error: Feature matrix or labels are empty after processing.")

    return X, y

def split_data(X, y, test_size=0.2):
    """Splits the dataset into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=42)

def train_knn(X_train, y_train, k=3):
    """Trains a kNN classifier with a specified value of k."""
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn

def evaluate_model(knn, X_train, y_train, X_test, y_test):
    """Evaluates the trained kNN model and prints performance metrics."""
    y_train_pred = knn.predict(X_train)
    y_test_pred = knn.predict(X_test)

    print("\nConfusion Matrix (Train Data):\n", confusion_matrix(y_train, y_train_pred))
    print("Classification Report (Train Data):\n", classification_report(y_train, y_train_pred))

    print("\nConfusion Matrix (Test Data):\n", confusion_matrix(y_test, y_test_pred))
    print("Classification Report (Test Data):\n", classification_report(y_test, y_test_pred))

    print(f"\nTraining Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

# Main Execution
file_path = "Judgment_Embeddings_InLegalBERT.xlsx"
df_project = load_dataset(file_path)
X_project, y_project = preprocess_data(df_project)
X_train, X_test, y_train, y_test = split_data(X_project, y_project)
knn_model = train_knn(X_train, y_train, k=3)
evaluate_model(knn_model, X_train, y_train, X_test, y_test)

