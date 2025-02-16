import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def calculate_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2score = r2_score(actual, predicted)
    
    return mse, rmse, mape, r2score

# Example Usage
actual = df["Price"]
predicted = df["Open"]
mse, rmse, mape, r2score = calculate_metrics(actual, predicted)

print(f"MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%, R²-Score: {r2score:.2f}")

