import pandas as pd
import numpy as np

file_name = r"/content/Lab Session Data.xlsx"
purchase_data = pd.read_excel(file_name, sheet_name="Purchase data")

purchase_data = purchase_data[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)', 'Payment (Rs)']]
purchase_data.columns = ['Candies', 'Mangoes', 'Milk_Packets', 'Payment']

X = purchase_data[['Candies', 'Mangoes', 'Milk_Packets']].to_numpy()
Y = purchase_data['Payment'].to_numpy().reshape(-1, 1)

X_augmented = np.column_stack([np.ones(X.shape[0]), X])

X_pseudo_inv = np.linalg.pinv(X_augmented)

model_vector = X_pseudo_inv @ Y

print("Model Vector (Intercept and Coefficients):")
print(model_vector.flatten())
