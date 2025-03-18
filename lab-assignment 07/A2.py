import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def generate_data(seed=42, num_points=100):
    np.random.seed(seed)
    X = np.random.rand(num_points, 2) * 10  # Generate features
    y = np.random.choice([0, 1], size=num_points)  # Generate labels
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def generate_test_grid(step=0.1, range_min=0, range_max=10):
    x_test = np.arange(range_min, range_max, step)
    y_test = np.arange(range_min, range_max, step)
    X_test_grid, Y_test_grid = np.meshgrid(x_test, y_test)
    return np.column_stack((X_test_grid.ravel(), Y_test_grid.ravel()))

def train_knn(X_train, y_train, k=3):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn

def plot_results(test_points, y_pred_test, X_train, y_train, k):
    plt.figure(figsize=(8, 6))
    plt.scatter(test_points[:, 0], test_points[:, 1], c=y_pred_test, cmap='bwr', alpha=0.3, marker='s')
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', edgecolors='k', marker='o')
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.title(f"kNN Classification (k={k})")
    plt.show()

# Main execution
X, y = generate_data()
X_train, X_test, y_train, y_test = split_data(X, y)
test_points = generate_test_grid()
knn_model = train_knn(X_train, y_train, k=3)
y_pred_test = knn_model.predict(test_points)
plot_results(test_points, y_pred_test, X_train, y_train, k=3)
