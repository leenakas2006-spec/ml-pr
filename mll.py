import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# 1. Loading and Exploring Data (تحميل واستكشاف البيانات)
insurance_dataset = pd.read_csv(r"C:\Users\user\Downloads\insurance.csv")

print("=== Dataset Description ===")
print(f"Number of records: {insurance_dataset.shape[0]}")
print(f"Number of features: {insurance_dataset.shape[1]}")
print(f"Features: {list(insurance_dataset.columns)}")
print("Target Variables: 'charges' (Regression), 'high_risk' (Classification)\n")

# 2. Data Preprocessing & Feature Engineering (تنظيف البيانات وهندسة الخصائص)
insurance_dataset = insurance_dataset.dropna()

# تعريف فئة "عالي الخطورة" بناءً على التدخين وكتلة الجسم
insurance_dataset['high_risk'] = ((insurance_dataset['smoker'] == 'yes') & (insurance_dataset['bmi'] > 30)).astype(int)

# تحويل البيانات النصية (الفئوية) لأرقام
categorical_cols = ['sex', 'smoker', 'region']
encoded_dataset = pd.get_dummies(insurance_dataset, columns=categorical_cols, drop_first=True)

# 3. Data Splitting & Fixing Data Leakage (تقسيم البيانات وحل مشكلة تسريب البيانات)

# --- تجهيز بيانات الانحدار (Regression) ---
# بنتوقع التكلفة باستخدام كل الخصائص المتاحة
features_regression = encoded_dataset.drop(columns=['charges', 'high_risk'])
target_charges = encoded_dataset['charges']

# --- تجهيز بيانات التصنيف (Classification) ---
# هون التريك! بنحذف 'bmi' و 'smoker_yes' عشان نمنع تسريب البيانات (Data Leakage)
features_classification = encoded_dataset.drop(columns=['charges', 'high_risk', 'bmi', 'smoker_yes'])
target_high_risk = encoded_dataset['high_risk']

# تقسيم البيانات لتدريب واختبار (للموديلين بشكل منفصل)
X_train_reg, X_test_reg, y_train_charges, y_test_charges = train_test_split(
    features_regression, target_charges, test_size=0.2, random_state=50
)

X_train_clf, X_test_clf, y_train_risk, y_test_risk = train_test_split(
    features_classification, target_high_risk, test_size=0.2, random_state=50
)

# 4. Feature Scaling (توحيد مقياس الأرقام لتسريع تدريب الموديل)
# لاحظ إننا شلنا الـ bmi من خصائص التصنيف لأنه انحذف بالخطوة السابقة
numerical_cols_reg = ['age', 'bmi', 'children']
numerical_cols_clf = ['age', 'children'] 

scaler_reg = StandardScaler()
X_train_reg[numerical_cols_reg] = scaler_reg.fit_transform(X_train_reg[numerical_cols_reg])
X_test_reg[numerical_cols_reg] = scaler_reg.transform(X_test_reg[numerical_cols_reg])

scaler_clf = StandardScaler()
X_train_clf[numerical_cols_clf] = scaler_clf.fit_transform(X_train_clf[numerical_cols_clf])
X_test_clf[numerical_cols_clf] = scaler_clf.transform(X_test_clf[numerical_cols_clf])

# 5. Model Training & Evaluation (تدريب النماذج وتقييمها)

# --- تدريب نموذج التوقع (Linear Regression) ---
linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train_reg, y_train_charges)
predictions_charges = linear_regression_model.predict(X_test_reg)

print("--- Regression Results (Predicting Charges) ---")
print("R2 Score:", r2_score(y_test_charges, predictions_charges))
print("RMSE:", np.sqrt(mean_squared_error(y_test_charges, predictions_charges)))
print("-" * 25 + "\n")

# --- تدريب نموذج التصنيف (Logistic Regression) ---
logistic_regression_model = LogisticRegression(max_iter=1000)
logistic_regression_model.fit(X_train_clf, y_train_risk)
predictions_risk = logistic_regression_model.predict(X_test_clf)

print("--- Classification Results (Predicting High Risk) ---")
print("Accuracy:", accuracy_score(y_test_risk, predictions_risk))
print("\nClassification Report:\n", classification_report(y_test_risk, predictions_risk))
