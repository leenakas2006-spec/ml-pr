# Health Insurance Cost Prediction & High-Risk Classification
# توقع تكاليف التأمين الصحي وتصنيف الحالات عالية الخطورة

## Overview / نظرة عامة
This project applies Machine Learning techniques to a real-world health insurance dataset (1,338 records). The primary objective is to understand how different patient attributes affect medical insurance costs and to build models that can automatically predict these costs and classify patient risk levels.

يطبق هذا المشروع تقنيات تعلم الآلة على مجموعة بيانات حقيقية للتأمين الصحي (1,338 سجل). الهدف الأساسي هو فهم كيف تؤثر خصائص المرضى المختلفة على تكاليف التأمين الطبي، وبناء نماذج قادرة على التنبؤ بهذه التكاليف وتصنيف مستويات الخطر للمرضى بشكل آلي.

## Project Goals / أهداف المشروع

1. **Regression Task (Predicting Charges):** 
   Predicting the continuous value of insurance charges based on features like age, sex, BMI, number of children, smoker status, and region.
   *مهمة الانحدار (توقع التكاليف):* توقع القيمة المستمرة لتكاليف التأمين بناءً على خصائص مثل العمر، الجنس، مؤشر كتلة الجسم، عدد الأطفال، حالة التدخين، والمنطقة.

2. **Classification Task (Flagging High-Risk Individuals):**
   Classifying individuals as "high-risk" (defined as a smoker with a BMI > 30).
   *مهمة التصنيف (تحديد الحالات عالية الخطورة):* تصنيف الأفراد كـ "حالة عالية الخطورة" (والتي تم تعريفها كشخص مدخن ومؤشر كتلة جسمه أعلى من 30).

## Technical Highlights & Data Leakage Fix / ملاحظات تقنية وحل مشكلة تسريب البيانات
A critical aspect of this project was identifying and resolving a Data Leakage issue in the classification task. Since the `high_risk` target variable was engineered directly from the `smoker` and `bmi` columns, including these features in the training set would allow the model to cheat by simply memorizing the rule, leading to an artificially high accuracy. 

To build a robust and scientifically accurate model, the `bmi` and `smoker` features were explicitly removed from the classification dataset before training. This forces the model to find underlying patterns using only the remaining demographic and regional data.

أحد الجوانب التقنية المهمة في هذا المشروع كان تحديد وحل مشكلة تسريب البيانات (Data Leakage) في مهمة التصنيف. نظراً لأن المتغير المستهدف `high_risk` تم بناؤه بالاعتماد المباشر على عمودي التدخين ومؤشر كتلة الجسم، فإن إبقاء هذه الأعمدة في بيانات التدريب كان سيسمح للنموذج بـ "الغش" واكتشاف القاعدة فقط بدلاً من تعلم أنماط حقيقية، مما يؤدي إلى دقة وهمية.
لبناء نموذج دقيق علمياً، تم حذف هذه الأعمدة من بيانات التصنيف قبل التدريب، مما يجبر النموذج على إيجاد الأنماط بالاعتماد على البيانات الديموغرافية المتبقية فقط.

## Methodologies / المنهجية المتبعة
* **Data Preprocessing:** Handling missing values, mapping categorical variables using One-Hot Encoding, and splitting the data into 80% training and 20% testing sets.
* **Feature Scaling:** Applied `StandardScaler` to numerical features (Age, BMI, Children) to optimize model performance.
* **Models Used:** 
  * `Linear Regression` for the continuous target (Charges).
  * `Logistic Regression` for the binary target (High-Risk).

* **تنظيف ومعالجة البيانات:** التعامل مع القيم المفقودة، تحويل المتغيرات الفئوية باستخدام One-Hot Encoding، وتقسيم البيانات إلى 80% للتدريب و 20% للاختبار.
* **توحيد القياس:** استخدام `StandardScaler` للخصائص الرقمية لتحسين أداء النماذج.
* **النماذج المستخدمة:** 
  * `Linear Regression` للهدف المستمر (التكاليف).
  * `Logistic Regression` للهدف الثنائي (عالي الخطورة).

## Technologies & Tools / التقنيات والأدوات
* Python
* pandas
* scikit-learn
* NumPy

## Team / فريق العمل
* Mohammed Suhail (محمد سهيل)
* Leena (لينا)
