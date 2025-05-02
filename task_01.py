import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

df = pd.read_csv('internship_candidates_final_numeric.csv')

X = df[["Experience", "Grade", "EnglishLevel", "Age", "EntryTestScore"]]
y = df["Accepted"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

import matplotlib.pyplot as plt

plt.scatter(X_test["Experience"], X_test["Grade"], c=y_pred, cmap='coolwarm', edgecolor='k', s=100)
plt.title("Logistic Regression Predictions")
plt.xlabel("Experience")
plt.ylabel("Grade")
plt.colorbar(label='Predicted Class')
plt.show()

plt.scatter(X_test["Experience"], X_test["EnglishLevel"], c=y_pred, cmap='coolwarm', edgecolor='k', s=100)
plt.title("Logistic Regression Predictions")
plt.xlabel("Experience")
plt.ylabel("English Level")
plt.colorbar(label='Predicted Class')
plt.show()

plt.scatter(X_test["Experience"], X_test["Age"], c=y_pred, cmap='coolwarm', edgecolor='k', s=100)
plt.title("Logistic Regression Predictions")
plt.xlabel("Experience")
plt.ylabel("Age")
plt.colorbar(label='Predicted Class')
plt.show()

plt.scatter(X_test["Experience"], X_test["EntryTestScore"], c=y_pred, cmap='coolwarm', edgecolor='k', s=100)
plt.title("Logistic Regression Predictions")
plt.xlabel("Experience")
plt.ylabel("Entry Test Score")
plt.colorbar(label='Predicted Class')
plt.show()

new_data = pd.DataFrame({
    "Experience": [5, 3, 2],
    "Grade": [85, 78, 90],
    "EnglishLevel": [8, 7, 9],
    "Age": [25, 30, 22],
    "EntryTestScore": [80, 75, 88]
})

predictions = model.predict(new_data)
print("Predictions for new data:", predictions)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))