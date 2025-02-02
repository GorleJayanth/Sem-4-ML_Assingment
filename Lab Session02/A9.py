import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from google.colab import files

file_path = r"/content/Lab Session Data.xlsx"
thyroid_data = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

first_two_rows = thyroid_data.iloc[:2]

numeric_columns = first_two_rows.select_dtypes(include=["int64", "float64"])

vector_1 = numeric_columns.iloc[0].values.reshape(1, -1)
vector_2 = numeric_columns.iloc[1].values.reshape(1, -1)

cosine_similarity_value = cosine_similarity(vector_1, vector_2)[0][0]

print("\nCosine Similarity between the first two observations:", round(cosine_similarity_value, 4))

if cosine_similarity_value > 0.8:
    print(" The vectors are highly similar.")
elif cosine_similarity_value > 0.5:
    print(" The vectors have moderate similarity.")
else:
    print(" The vectors are not very similar.")
