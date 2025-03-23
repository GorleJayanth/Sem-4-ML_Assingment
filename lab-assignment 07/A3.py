import pandas as pd
import numpy as np
import sys

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer

# Load dataset with error handling
def load_dataset(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name=0)  # Load first sheet
        df.columns = df.columns.str.strip()  # Clean column names
        if 'Label' not in df.columns:
            raise ValueError("'Label' column not found in dataset.")
        return df
    except Exception as e:
        print("Error loading file:", e)
        sys.exit()

# Handle missing values
def preprocess_data(df):
    imputer = SimpleImputer(strategy='mean')
    df[df.columns] = imputer.fit_transform(df)
    X = df.drop(columns=['Label'])
    y = df['Label'].astype(int)  # Ensure Label is integer
    return X, y

# Train and evaluate SVM classifier
def train_svm(X_train, X_test, y_train, y_test):
    print("Training SVM classifier...")
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print("\nSVM Classifier Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

# Main function
def main():
    file_path = "Judgment_Embeddings_InLegalBERT.xlsx"
    df = load_dataset(file_path)
    X, y = preprocess_data(df)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Train SVM classifier
    train_svm(X_train, X_test, y_train, y_test)

# Run the script
if __name__ == "__main__":
    main()
