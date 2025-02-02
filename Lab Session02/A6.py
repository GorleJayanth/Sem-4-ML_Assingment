import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files

file_path = r"/content/Lab Session Data.xlsx"
thyroid_data = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

missing_before_imputation = thyroid_data.isnull().sum()
print("\nMissing Values Before Imputation:\n", missing_before_imputation)

numeric_columns = thyroid_data.select_dtypes(include=["int64", "float64"]).columns
categorical_columns = thyroid_data.select_dtypes(include=["object"]).columns

print("\nNumeric Columns:", numeric_columns)
print("\nCategorical Columns:", categorical_columns)

plt.figure(figsize=(12, 6))
sns.boxplot(data=thyroid_data[numeric_columns], palette="viridis")
plt.xticks(rotation=45)
plt.title("Boxplot to Detect Outliers")
plt.show()

for column in numeric_columns:
    if thyroid_data[column].isnull().any():
        skew_value = thyroid_data[column].skew()
        if abs(skew_value) < 1:
            thyroid_data[column].fillna(thyroid_data[column].mean(), inplace=True)  
        else:
            thyroid_data[column].fillna(thyroid_data[column].median(), inplace=True)  

missing_after_numeric_imputation = thyroid_data.isnull().sum()
print("\nMissing Values After Numeric Imputation:\n", missing_after_numeric_imputation)

for column in categorical_columns:
    if thyroid_data[column].isnull().any():
        thyroid_data[column].fillna(thyroid_data[column].mode()[0], inplace=True)  

missing_after_categorical_imputation = thyroid_data.isnull().sum()
print("\nMissing Values After Categorical Imputation:\n", missing_after_categorical_imputation)

imputed_file_path = "Imputed_Thyroid_Data.xlsx"
thyroid_data.to_excel(imputed_file_path, index=False)

files.download(imputed_file_path)
