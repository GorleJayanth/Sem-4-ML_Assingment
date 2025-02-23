import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# Sample data (Replace with actual dataset values)
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])  # Feature
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
