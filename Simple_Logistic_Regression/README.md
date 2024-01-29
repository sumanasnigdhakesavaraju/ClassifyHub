### Project Title: Simple Logistic Regression on Iris Dataset

## Overview:
This project explores the implementation of Simple Logistic Regression on the famous Iris dataset. The goal is to provide a clear understanding of the classification process, focusing on the Iris dataset's features and target classes.

## Prerequisites:
Ensure the following prerequisites for successful notebook execution:
- **Python Version:** 3.9 or above
- **Datasets:** Download the Iris dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/iris) and follow the loading steps provided in the notebook.
- **Libraries:** All required libraries are imported within the notebook.


## Simple Logistic Regression 
### What is Simple Logistic Regression?
Simple Logistic Regression is a statistical method used for binary classification tasks, where the outcome variable (dependent variable) has two possible classes. It models the relationship between a single independent variable (feature) and the probability of the binary outcome. The logistic regression model employs the logistic function to transform a linear combination of the input features into probabilities, providing a predicted probability of belonging to a particular class. The decision boundary is determined based on a chosen probability threshold. This method is particularly useful for scenarios where the target variable is binary, such as predicting whether an email is spam or not spam.

### Formula:
The Simple Logistic Regression formula is expressed as:

\[ P(Y=1|X) = \frac{1}{1 + e^{-(b_0 + b_1X)}} \]

Where:
- \( P(Y=1|X) \) is the probability that the output \( Y \) is 1 given input \( X \).
- \( b_0 \) is the intercept.
- \( b_1 \) is the coefficient of the feature \( X \).
- \( e \) is the base of the natural logarithm.

## Results:
The accuracy of the Simple Logistic Regression model on the Iris dataset is close to 96%.

---

Feel free to explore the provided notebook for a detailed walkthrough of the Simple Logistic Regression implementation on the Iris dataset. For any questions or clarifications, please refer to the notebook documentation.
