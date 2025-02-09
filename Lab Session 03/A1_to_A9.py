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

#A2
feature = "feature_0" 
plt.hist(df[feature])
plt.xlabel(feature)
plt.ylabel("Frequency")
plt.title(f"Histogram of {feature}")
plt.show()
print(f"Mean of {feature}: {df[feature].mean()}")
print(f"Variance of {feature}: {df[feature].var()}")

# A3
vec1 = X.iloc[0].values
vec2 = X.iloc[1].values
minkowski_distances = []
for r in range(1, 11):
    dist = minkowski(vec1, vec2, r) 
    minkowski_distances.append(dist)
    print(f"Minkowski distance for r={r} is {dist}")
plt.plot(range(1, 11), minkowski_distances, marker="o")
plt.xlabel("Minkowski Order (r)")
plt.ylabel("Distance")
plt.title("Minkowski Distance vs r")
plt.show()

# A4
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

# A5
kNN = KNeighborsClassifier(n_neighbors=3)
kNN.fit(X_train, Y_train)

# A6
accuracy = kNN.score(X_test, Y_test)
print("k-NN Accuracy:", accuracy)

# A7: 
predictions = kNN.predict(X_test)
print("Predictions:", predictions)

# A8:
k_values = range(1, 12)
accuracies = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    acc = knn.score(X_test, Y_test)
    accuracies.append(acc)
plt.plot(k_values, accuracies, marker="o")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("k vs Accuracy")
plt.show()

#A9:
y_pred = kNN.predict(X_train)
conf_matrix = confusion_matrix(Y_train, y_pred)
print(f"Confusion Matrix(training):\n{conf_matrix}")
print(f"Classification Report(training):\n{classification_report(Y_train, y_pred)}")
Y_pred = kNN.predict(X_test)
conf_matrix = confusion_matrix(Y_test, Y_pred)
print(f"Confusion Matrix(test):\n{conf_matrix}")
print(f"Classification Report(test):\n{classification_report(Y_test, Y_pred)}")
