import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

file_path = r"/content/Lab Session Data.xlsx"
thyroid_data = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

print("Preview of the data:")
print(thyroid_data.head())

print("\nData Types of Columns:\n", thyroid_data.dtypes)

categorical_columns = thyroid_data.select_dtypes(include=['object']).columns
print("\nCategorical Columns Detected:\n", categorical_columns)

numeric_columns = thyroid_data.select_dtypes(include=['int64', 'float64']).columns
print("\nNumeric Variables - Range (Min and Max):\n", thyroid_data[numeric_columns].agg(['min', 'max']))

missing_values_count = thyroid_data.isnull().sum()
print("\nCount of Missing Values in Each Column:\n", missing_values_count)

plt.figure(figsize=(14, 7))
sns.boxplot(data=thyroid_data[numeric_columns], orient='h', palette='coolwarm')
plt.title("Boxplots to Visualize Outliers in Numeric Data")
plt.show()

mean_vals = thyroid_data[numeric_columns].mean()
std_vals = thyroid_data[numeric_columns].std()

print("\nMean Values of Numeric Variables:\n", mean_vals)
print("\nStandard Deviation Values of Numeric Variables:\n", std_vals)
