import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Sample data (Replace with actual dataset values)
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]])  # Features

# Splitting dataset into train and test sets
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# A6: Perform K-Means Clustering for Different k Values with Error Handling
def perform_kmeans_for_different_k(X_train, k_values):
    """Perform K-Means clustering for different k values and evaluate scores."""
    silhouette_scores = []
    ch_scores = []
    db_scores = []

    for k in k_values:
        if k >= len(X_train):  # Prevent invalid k values
            print(f"Skipping k={k} (more clusters than data points).")
            continue

        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans.fit(X_train)
            labels = kmeans.labels_

            if len(set(labels)) < 2:  # Ensure at least two clusters exist
                print(f"Skipping k={k} (less than 2 clusters found).")
                continue

            silhouette = silhouette_score(X_train, labels)
            ch_score = calinski_harabasz_score(X_train, labels)
            db_index = davies_bouldin_score(X_train, labels)

            silhouette_scores.append(silhouette)
            ch_scores.append(ch_score)
            db_scores.append(db_index)

        except Exception as e:
            print(f"Error for k={k}: {e}")

    return silhouette_scores, ch_scores, db_scores, list(k_values)[:len(silhouette_scores)]

# Define k values and execute function
k_values = range(2, 10)
silhouette_scores, ch_scores, db_scores, valid_k_values = perform_kmeans_for_different_k(X_train, k_values)

# Plot clustering evaluation metrics
plt.figure(figsize=(10, 5))
plt.plot(valid_k_values, silhouette_scores, marker='o', label='Silhouette Score')
plt.plot(valid_k_values, ch_scores, marker='s', label='Calinski-Harabasz Score')
plt.plot(valid_k_values, db_scores, marker='^', label='Davies-Bouldin Index')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Score')
plt.title('K-Means Clustering Evaluation')
plt.legend()
plt.show()

# Output results for A6
print("A6: Clustering Evaluation for Different k Values")
for i, k in enumerate(valid_k_values):
    print(f"k={k}: Silhouette={silhouette_scores[i]:.4f}, CH={ch_scores[i]:.4f}, DB={db_scores[i]:.4f}")
