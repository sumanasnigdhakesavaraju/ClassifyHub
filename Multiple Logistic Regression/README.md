### Project Title: Multiple Logistic Regression on Iris Dataset

## Overview:
This project delves into the implementation of Multiple Logistic Regression on the renowned Iris dataset. The primary objective is to elucidate the intricacies of the classification process, emphasizing the utilization of multiple features to predict the target classes.

## Prerequisites:
Ensure the following prerequisites for seamless execution of the notebook:
- **Python Version:** 3.9 or above
- **Datasets:** Download the Iris dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/iris) and adhere to the loading steps outlined in the notebook.
- **Libraries:** All essential libraries are imported within the notebook.

## Multiple Logistic Regression
### What is Multiple Logistic Regression?
Multiple Logistic Regression extends the concept of Simple Logistic Regression to accommodate scenarios with more than two classes. It is a statistical method employed for multiclass classification tasks. The model formulates the relationship between multiple independent variables (features) and the probability of belonging to each class. The logistic function transforms a linear combination of input features into probabilities for each class, facilitating a comprehensive prediction across multiple categories.

### Formula:
The Multiple Logistic Regression formula for predicting the probability of class \( k \) given input \( X \) is expressed as:

\[ P(Y=k|X) = \frac{e^{b_{0k} + b_{1k}X_1 + ... + b_{pk}X_p}}{1 + e^{b_{01} + b_{11}X_1 + ... + b_{p1}X_p} + ... + e^{b_{0K} + b_{1K}X_1 + ... + b_{pK}X_p}} \]

Where:
- \( P(Y=k|X) \) is the probability that the output \( Y \) is class \( k \) given input \( X \).
- \( b_{0k}, b_{1k}, ..., b_{pk} \) are the intercept and coefficients for class \( k \).
- \( X_1, X_2, ..., X_p \) are the features.
- \( K \) is the number of classes.

## Results:
The Multiple Logistic Regression model on the Iris dataset achieves an accuracy exceeding 90%, showcasing its efficacy in handling multiclass classification tasks.

---

Feel free to explore the provided notebook for an in-depth exploration of the Multiple Logistic Regression implementation on the Iris dataset. For any inquiries or clarifications, please consult the notebook documentation.
