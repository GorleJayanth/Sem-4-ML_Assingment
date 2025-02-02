import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

file_path = r"/content/Lab Session Data.xlsx"
purchase_data = pd.read_excel(file_path, sheet_name="Purchase data")

purchase_data['Customer'] = purchase_data['Payment (Rs)'].apply(lambda payment: 'RICH' if payment > 200 else 'POOR')

X_data = purchase_data[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']]
Y_data = purchase_data['Customer']

X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.5, random_state=42)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, Y_train)

predictions = knn_model.predict(X_test)

print("Classification Report:")
print(classification_report(Y_test, predictions, zero_division=0))

