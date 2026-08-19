import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"C:\Users\user\Downloads\insurance.csv")

# 1. 

print("=== Dataset Description ===")
print(f"Number of records: {df.shape[0]}")
print(f"Number of features: {df.shape[1]}")
print(f"Features: {list(df.columns)}")
print("Target Variables: 'charges' (Regression), 'high_risk' (Classification)\n")

# 2. 
df = df.dropna()
df['high_risk'] = ((df['smoker'] == 'yes') & (df['bmi'] > 30)).astype(int)
categorical_cols = ['sex', 'smoker', 'region']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 3.
X = df_encoded.drop(columns=['charges', 'high_risk'])
y_reg = df_encoded['charges']
y_clf = df_encoded['high_risk']
X_train, X_test, y_train_reg, y_test_reg, y_train_clf, y_test_clf = train_test_split(
    X, y_reg, y_clf, test_size=0.2, random_state=50
)

# 4.
numerical_cols = ['age', 'bmi', 'children']
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# 5.
reg_model = LinearRegression()
reg_model.fit(X_train, y_train_reg)
y_pred_reg = reg_model.predict(X_test)
print("--- Regression Results ---")
print("R2 Score:", r2_score(y_test_reg, y_pred_reg))
print("RMSE:", np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)))
print("-" * 25 + "\n")
clf_model = LogisticRegression(max_iter=1000)
clf_model.fit(X_train, y_train_clf)
y_pred_clf = clf_model.predict(X_test)
print("--- Classification Results ---")
print("Accuracy:", accuracy_score(y_test_clf, y_pred_clf))
print("\nClassification Report:\n", classification_report(y_test_clf, y_pred_clf))


