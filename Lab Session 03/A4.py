import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files
from scipy.spatial.distance import minkowski
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

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

knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(X_train, y_train)
score = knn.score(X_test, y_test)
print("Model R² Score:", score)


