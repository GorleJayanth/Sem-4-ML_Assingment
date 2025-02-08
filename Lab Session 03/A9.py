import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

uploaded = files.upload()
file_name = list(uploaded.keys())[0]
df = pd.read_excel(pd.ExcelFile(file_name), sheet_name="IRCTC Stock Price")

closing_price_col = 'Close' if 'Close' in df.columns else 'Price'
if closing_price_col == 'Price':
    print(f"Warning: 'Close' column not found. Using '{closing_price_col}' instead.")

X, y = df[['Open', 'Low']], (df[closing_price_col] > df['Open']).astype(int)
df.dropna(subset=['Open', 'Low', closing_price_col], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
y_pred = knn.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'High'], yticklabels=['Low', 'High'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

print("Classification Report:\n", classification_report(y_test, y_pred))
