import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from scipy.spatial.distance import minkowski

df = pd.read_excel("Judgment_Embeddings_InLegalBERT.xlsx")  
X = df.drop(columns=["Label"])  
Y = df["Label"]  

feature = "feature_0" 
plt.hist(df[feature])
plt.xlabel(feature)
plt.ylabel("Frequency")
plt.title(f"Histogram of {feature}")
plt.show()
print(f"Mean of {feature}: {df[feature].mean()}")
print(f"Variance of {feature}: {df[feature].var()}")
