y_pred = kNN.predict(X_train)
conf_matrix = confusion_matrix(Y_train, y_pred)
print(f"Confusion Matrix(training):\n{conf_matrix}")
print(f"Classification Report(training):\n{classification_report(Y_train, y_pred)}")
Y_pred = kNN.predict(X_test)
conf_matrix = confusion_matrix(Y_test, Y_pred)
print(f"Confusion Matrix(test):\n{conf_matrix}")
print(f"Classification Report(test):\n{classification_report(Y_test, Y_pred)}")
