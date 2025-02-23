import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.cluster import KMeans

# Sample data (Replace with actual dataset values)
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11]])  # Features
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])  # Target

# Splitting dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# A1: Train Linear Regression Model
def train_linear_regression(X_train, y_train):
    """Train a linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def make_predictions(model, X):
    """Make predictions using the trained model."""
    return model.predict(X)

# Train model and make predictions for A1
model = train_linear_regression(X_train, y_train)
y_train_pred = make_predictions(model, X_train)
y_test_pred = make_predictions(model, X_test)

# A2: Evaluate Model
def evaluate_model(y_true, y_pred):
    """Calculate and return evaluation metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mape, r2

# Compute evaluation metrics for A2
train_mse, train_rmse, train_mape, train_r2 = evaluate_model(y_train, y_train_pred)
test_mse, test_rmse, test_mape, test_r2 = evaluate_model(y_test, y_test_pred)

# Output results for A2
print("A2: Model Evaluation Results")
print(f"Train -> MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, MAPE: {train_mape:.4f}, R2: {train_r2:.4f}")
print(f"Test -> MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAPE: {test_mape:.4f}, R2: {test_r2:.4f}")

# A3: Train and Evaluate Model with Multiple Features
def train_and_evaluate_multi_feature(X_train, X_test, y_train, y_test):
    """Train a linear regression model with multiple features and evaluate it."""
    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mse, train_rmse, train_mape, train_r2 = evaluate_model(y_train, y_train_pred)
    test_mse, test_rmse, test_mape, test_r2 = evaluate_model(y_test, y_test_pred)

    return train_mse, train_rmse, train_mape, train_r2, test_mse, test_rmse, test_mape, test_r2

# Train and evaluate with multiple features
train_mse, train_rmse, train_mape, train_r2, test_mse, test_rmse, test_mape, test_r2 = train_and_evaluate_multi_feature(X_train, X_test, y_train, y_test)

# Output results for A3
print("A3: Multi-Feature Model Evaluation Results")
print(f"Train -> MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}, MAPE: {train_mape:.4f}, R2: {train_r2:.4f}")
print(f"Test -> MSE: {test_mse:.4f}, RMSE: {test_rmse:.4f}, MAPE: {test_mape:.4f}, R2: {test_r2:.4f}")

# A4: Perform K-Means Clustering
def perform_kmeans_clustering(X_train, n_clusters=2):
    """Perform K-Means clustering and return labels and cluster centers."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(X_train)
    return kmeans.labels_, kmeans.cluster_centers_

# Execute K-Means Clustering for A4
cluster_labels, cluster_centers = perform_kmeans_clustering(X_train, n_clusters=2)

# Output results for A4
print("A4: K-Means Clustering Results")
print(f"Cluster Labels: {cluster_labels}")
print(f"Cluster Centers: {cluster_centers}")
