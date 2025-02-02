import pandas as pd
import numpy as np

file = pd.read_excel(r"/content/Lab Session Data.xlsx", sheet_name="Purchase data")

product_data = file[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']]
payment_data = file['Payment (Rs)']

A = np.array(product_data)
C = np.array(payment_data).reshape(-1, 1)

print(f"A = {A}")
print(f"C = {C}")

dimensionality = A.shape[1]
vector_space = A.shape[0]
print(f"the dimensionality of the vector space is = {dimensionality}")
print(f"the number of vectors in the vector space is = {vector_space}")

rank_A = np.linalg.matrix_rank(A)
print(f"the rank of the matrix A is = {rank_A}")

pseudo_inverse = np.linalg.pinv(A)
print(f"the pseudo inverse of matrix A is = \n{pseudo_inverse}")

cost = pseudo_inverse @ C
print(f"the cost of each product that is available for sale is  = {cost.flatten()}")
