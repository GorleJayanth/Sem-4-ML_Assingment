import numpy as np
import pandas as pd

# Replace 'your_file.xlsx' with the actual file path if it's stored in a known location
file_name = 'Lab Session Data.xlsx'  

xls = pd.ExcelFile(file_name)
df = xls.parse('IRCTC Stock Price')

def classify_price(price):
    return "Low" if price < 2000 else "Mid" if price < 2500 else "High"

df['Price Range'] = df['Price'].apply(classify_price)

columns = ["Price", "Open", "High", "Low", "Volume", "Chg%"]

def transform_volume(value):
    if isinstance(value, str):
        if 'K' in value:
            return float(value.replace('K', '')) * 1e3
        elif 'M' in value:
            return float(value.replace('M', '')) * 1e6
        try:
            return float(value)
        except ValueError:
            return np.nan
    return value

df['Volume'] = df['Volume'].apply(transform_volume)

grouped = df.groupby("Price Range")[columns]
centroids = grouped.mean()
spreads = grouped.std()

def calculate_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

categories = centroids.index.tolist()
distances = {}

for i in range(len(categories)):
    for j in range(i + 1, len(categories)):
        distances[(categories[i], categories[j])] = calculate_distance(centroids.loc[categories[i]], centroids.loc[categories[j]])

print("\n Class Centroids (Mean Values) ")
print(centroids)

print("\n Intraclass Spread (Standard Deviation) ")
print(spreads)

print("\n Interclass Distances ")
for (cls1, cls2), dist in distances.items():
    print(f"Distance between {cls1} and {cls2}: {dist:.2f}")
