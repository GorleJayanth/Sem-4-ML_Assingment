import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from google.colab import files

uploaded = files.upload()
file_name = next(iter(uploaded))
xls = pd.ExcelFile(file_name)

df = pd.read_excel(xls, sheet_name="IRCTC Stock Price")
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df.dropna(subset=["Price"], inplace=True)

hist_values, bin_edges = np.histogram(df["Price"], bins=10)

plt.figure(figsize=(8, 5))
plt.hist(df["Price"], bins=10, edgecolor='black', alpha=0.7)
plt.xlabel("Stock Price")
plt.ylabel("Frequency")
plt.title("Distribution of IRCTC Stock Prices")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

mean_val = df["Price"].mean()
var_val = df["Price"].var()

print(f"Average Stock Price: {mean_val}")
print(f"Price Variance: {var_val}")
