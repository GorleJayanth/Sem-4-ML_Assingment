import pandas as pd
import numpy as np
from google.colab import files

file_path = r"/content/Lab Session Data.xlsx"
thyroid_data = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

first_two_rows = thyroid_data.iloc[:2]

binary_columns = [
    column
    for column in thyroid_data.columns
    if thyroid_data[column].dropna().isin([0, 1, 0.0, 1.0]).all() and thyroid_data[column].nunique() <= 2
]

binary_data_from_vectors = first_two_rows[binary_columns]

vector_1 = binary_data_from_vectors.iloc[0].values
vector_2 = binary_data_from_vectors.iloc[1].values

f11 = np.sum((vector_1 == 1) & (vector_2 == 1))  
f00 = np.sum((vector_1 == 0) & (vector_2 == 0))  
f10 = np.sum((vector_1 == 1) & (vector_2 == 0))  
f01 = np.sum((vector_1 == 0) & (vector_2 == 1))  

JC_value = f11 / (f01 + f10 + f11) if (f01 + f10 + f11) != 0 else 0

SMC_value = (f11 + f00) / (f00 + f01 + f10 + f11) if (f00 + f01 + f10 + f11) != 0 else 0

print("\nConsidered Binary Columns:", binary_columns)
print(f"\nJaccard Coefficient (JC): {round(JC_value, 4)}")
print(f"Simple Matching Coefficient (SMC): {round(SMC_value, 4)}")

# Change 'JC' to 'JC_value' in the if condition
if JC_value < SMC_value:  
    print("\n SMC is higher than JC because it accounts for both matches (1,1 and 0,0).")
    print(" JC is more relevant when we focus on the presence of features (1s).")
else:
    print("\n JC and SMC are close, suggesting similarity in the feature vectors.")
