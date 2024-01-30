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

# Multiple Logistic Regression
multiple_logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=10000)
multiple_logreg.fit(X_train, y_train)
y_pred_multiple = multiple_logreg.predict(X_test)

# Create a confusion matrix
conf_matrix_mlr = confusion_matrix(y_test, y_pred_multiple)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(4, 4))
sns.heatmap(conf_matrix_mlr, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Multiple Logistic Regression')
plt.xticks(ticks=[0.5, 1.5, 2.5], labels=['Setosa', 'Versicolor', 'Virginica'])
plt.yticks(ticks=[0.5, 1.5, 2.5], labels=['Setosa', 'Versicolor', 'Virginica'])
plt.show()

