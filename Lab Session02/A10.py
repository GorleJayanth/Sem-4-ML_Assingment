import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from google.colab import files

file_path = r"/content/Lab Session Data.xlsx"
thyroid_data = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

top_20_vectors = thyroid_data.head(20)

binary_columns = []

for col in thyroid_data.columns:
    if thyroid_data[col].apply(lambda x: x in {0, 1}).all() and pd.api.types.is_numeric_dtype(thyroid_data[col]):
        binary_columns.append(col)

binary_subset = top_20_vectors[binary_columns]

binary_matrix = binary_subset.to_numpy()

numeric_subset = top_20_vectors.select_dtypes(include=["int64", "float64"]).to_numpy()

def compute_similarity_matrices(data_matrix):
    num_rows = data_matrix.shape[0]
    
    jc_sim_matrix = np.zeros((num_rows, num_rows))
    smc_sim_matrix = np.zeros((num_rows, num_rows))

    for i in range(num_rows):
        for j in range(num_rows):
            f11 = np.sum((data_matrix[i] == 1) & (data_matrix[j] == 1))
            f00 = np.sum((data_matrix[i] == 0) & (data_matrix[j] == 0))
            f10 = np.sum((data_matrix[i] == 1) & (data_matrix[j] == 0))
            f01 = np.sum((data_matrix[i] == 0) & (data_matrix[j] == 1))

            jc_sim_matrix[i, j] = f11 / (f01 + f10 + f11) if (f01 + f10 + f11) > 0 else 0

            smc_sim_matrix[i, j] = (f11 + f00) / (f00 + f01 + f10 + f11) if (f00 + f01 + f10 + f11) > 0 else 0

    return jc_sim_matrix, smc_sim_matrix

jaccard_matrix, smc_matrix = compute_similarity_matrices(binary_matrix)

cosine_sim_matrix = cosine_similarity(numeric_subset)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

sns.heatmap(jaccard_matrix, annot=True, cmap="YlGnBu", ax=axes[0])
axes[0].set_title("Jaccard Similarity")

sns.heatmap(smc_matrix, annot=True, cmap="YlGnBu", ax=axes[1])
axes[1].set_title("Simple Matching Coefficient")

sns.heatmap(cosine_sim_matrix, annot=True, cmap="YlGnBu", ax=axes[2])
axes[2].set_title("Cosine Similarity")

plt.tight_layout()
plt.show()
