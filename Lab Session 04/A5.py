import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def generate_synthetic_data():
    """Generates synthetic dataset with 100 points in 2D space."""
    np.random.seed(42)
    X = np.random.rand(100, 2) * 10  # Feature space
    y = np.random.choice([0, 1], size=100)  # Binary labels
    return X, y

def split_data(X, y, test_size=0.2):
    """Splits data into training and testing sets."""
    return train_test_split(X, y, test_size=test_size, random_state=42)

def generate_test_grid():
    """Generates a test grid for classification visualization."""
    x_test = np.arange(0, 10, 0.1)
    y_test = np.arange(0, 10, 0.1)
    X_test_grid, Y_test_grid = np.meshgrid(x_test, y_test)
    return np.column_stack((X_test_grid.ravel(), Y_test_grid.ravel()))

def knn_classification(X_train, y_train, test_points, k):
    """Trains a kNN model and predicts on test grid."""
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn.predict(test_points)

def plot_results(X_train, y_train, test_points, y_pred_test, k):
    """Plots the classification results."""
    plt.figure(figsize=(8, 6))
    plt.scatter(test_points[:, 0], test_points[:, 1], c=y_pred_test, cmap='bwr', alpha=0.3, marker='s')
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', edgecolors='k', marker='o')
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.title(f"kNN Classification (k={k})")
    plt.show()

# Main Execution
X, y = generate_synthetic_data()
X_train, X_test, y_train, y_test = split_data(X, y)
test_points = generate_test_grid()

for k in [1, 3, 5, 10, 15]:
    y_pred_test = knn_classification(X_train, y_train, test_points, k)
    plot_results(X_train, y_train, test_points, y_pred_test, k)

