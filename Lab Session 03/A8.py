import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files
from scipy.spatial.distance import minkowski
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

uploaded = files.upload()
file_name = list(uploaded.keys())[0]
xls = pd.ExcelFile(file_name)

df = pd.read_excel(xls, sheet_name="IRCTC Stock Price")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
df["Low"] = pd.to_numeric(df["Low"], errors="coerce")
df["High"] = pd.to_numeric(df["High"], errors="coerce")
df.dropna(subset=["Price", "Open", "Low", "High"], inplace=True)

X = df[['Open', 'Low']]
y = df['High']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

k_values = range(1, 12)
scores = []

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    score = knn.score(X_test, y_test)
    scores.append(score)
    print(f"k={k}, R² Score: {score}")

plt.figure(figsize=(8, 5))
plt.plot(k_values, scores, marker='o', linestyle='dashed', color='b')
plt.xlabel('k Value')
plt.ylabel('R² Score')
plt.title('kNN Accuracy for Different k Values')
plt.grid(True)
plt.show()
