# Health Insurance Cost Prediction & High-Risk Classification

## Overview
This project applies Machine Learning techniques to a real-world health insurance dataset (1,338 records). The primary objective is to understand how different patient attributes affect medical insurance costs and to build models that can automatically predict these costs and classify patient risk levels.

## Project Goals

1. **Regression Task (Predicting Charges):** 
   Predicting the continuous value of insurance charges based on features like age, sex, BMI, number of children, smoker status, and region.

2. **Classification Task (Flagging High-Risk Individuals):**
   Classifying individuals as "high-risk" (defined as a smoker with a BMI > 30).

## Technical Highlights & Handling Data Leakage
A critical aspect of this project was identifying and resolving a Data Leakage issue in the classification task. Since the `high_risk` target variable was engineered directly from the `smoker` and `bmi` columns, including these features in the training set would allow the model to cheat by simply memorizing the rule, leading to an artificially high accuracy. 

To build a robust and scientifically accurate model, the `bmi` and `smoker` features were explicitly removed from the classification dataset before training. This forces the model to find underlying patterns using only the remaining demographic and regional data, rather than relying on a pre-programmed rule.

## Methodologies
* **Data Preprocessing:** Handling missing values, mapping categorical variables using One-Hot Encoding, and splitting the data into 80% training and 20% testing sets.
* **Feature Scaling:** Applied `StandardScaler` to numerical features (Age, BMI, Children) to optimize model performance.
* **Models Used:** 
  * `Linear Regression` for the continuous target (Charges).
  * `Logistic Regression` for the binary target (High-Risk).

## Technologies & Tools
* Python
* pandas
* scikit-learn
* NumPy

## Looking Forward
This project is just the beginning of our journey in Artificial Intelligence and Data Science, and the upcoming projects will be even more advanced and impactful.

## Team
* Mohammed 
* Leena
