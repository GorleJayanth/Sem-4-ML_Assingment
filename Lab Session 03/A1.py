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

#A1
class1 = df[df["Label"] == 0]
class2 = df[df["Label"] == 1]
mean1 = class1.mean(axis=0)
print(f"Mean of class 1:\n {mean1}")
mean2 = class2.mean(axis=0)
print(f"Mean of class 2:\n{mean2}")
spread1 = class1.std(axis=0)
spread2 = class2.std(axis=0)
print(f"Spread of class 1:\n{spread1}")
print(f"Spread of class 2:\n{spread2}")
interclass_distance = np.linalg.norm(mean1 - mean2)
print(f"Interclass distance is {interclass_distance}")
