import numpy as np
import matplotlib.pyplot as plt

def generate_training_data():
    np.random.seed(42)
    X_train = np.random.uniform([1, 1], [10, 10], (20, 2))
    y_train = np.random.randint(0, 2, 20)

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', edgecolors='k')
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.title("Training Data - Scatter Plot")
    plt.show()
    
    return X_train, y_train

# Example Usage
X_train, y_train = generate_training_data()

