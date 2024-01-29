# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Load Iris dataset
iris = load_iris()
X_i = iris.data  # Features
y_i = iris.target  # Target classes (0, 1, 2)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_i, y_i, test_size=0.3, random_state=42)

# Initialize and train Logistic Regression model
logistic_reg = LogisticRegression(solver='lbfgs', multi_class='ovr')  # 'ovr' for binary classification
logistic_reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred_simple = logistic_reg.predict(X_test)

# Plot the actual and predicted class labels
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', label='Actual')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred_simple, marker='*', label='Predicted')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()

# Create a confusion matrix
conf_matrix_slr = confusion_matrix(y_test, y_pred_simple)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(4, 4))
sns.heatmap(conf_matrix_slr, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Simple Logistic Regression')
plt.xticks(ticks=[0.5, 1.5, 2.5], labels=['Setosa', 'Versicolor', 'Virginica'])
plt.yticks(ticks=[0.5, 1.5, 2.5], labels=['Setosa', 'Versicolor', 'Virginica'])
plt.show()

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred_simple)
print("Accuracy:", accuracy)
