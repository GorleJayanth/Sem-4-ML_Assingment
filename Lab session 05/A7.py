import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Sample data (Replace with actual dataset values)
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]])  # Features

# Splitting dataset into train and test sets
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# A7: Determine Optimal k using Elbow Method with Error Handling
def determine_optimal_k(X_train, max_k):
    """Use the Elbow method to determine the optimal k for K-Means clustering."""
    distortions = []
    k_values = list(range(2, min(max_k, len(X_train)) + 1))  # Ensure k does not exceed available data points

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)  # Explicit n_init
        kmeans.fit(X_train)
        distortions.append(kmeans.inertia_)  # Inertia is the sum of squared distances to cluster centers

    return distortions, k_values

# Define maximum k value and compute distortions
max_k = 10
distortions, valid_k_values = determine_optimal_k(X_train, max_k)

# Plot the Elbow Method
plt.figure(figsize=(10, 5))
plt.plot(valid_k_values, distortions, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Distortion (Inertia)')
plt.title('Elbow Method for Optimal k')
plt.xticks(valid_k_values)
plt.grid(True)
plt.show()

# Output results for A7
print("A7: Elbow Method Results")
for i, k in enumerate(valid_k_values):
    print(f"k={k}: Distortion={distortions[i]:.4f}")
