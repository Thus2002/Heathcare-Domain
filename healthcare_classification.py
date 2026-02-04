# -----------------------------------
# Healthcare Domain Mini Project
# Disease Classification
# -----------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------------
# 1. Load Dataset
# -----------------------------------
print("Loading healthcare dataset...")
data = pd.read_csv("healthcare_dataset.csv")
print("Dataset loaded successfully")
print("Dataset shape:", data.shape)

# -----------------------------------
# 2. Feature and Target Separation
# -----------------------------------
X = data.drop("Disease", axis=1)
y = data["Disease"]

# -----------------------------------
# 3. Train-Test Split
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------------
# 4. Train Classification Model
# -----------------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------------
# 5. Model Evaluation
# -----------------------------------
y_pred = model.predict(X_test)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------------
# 6. Disease Prediction for New Patient
# -----------------------------------
# Input order: Age, BloodPressure, Cholesterol, BMI, Glucose
sample_patient = [[45, 130, 220, 28, 120]]

prediction = model.predict(sample_patient)

if prediction[0] == 1:
  print("\nPredicted Result: Disease Present")
else:
  print("\nPredicted Result: No Disease Present")
  
    print("\nPredicted Result: No Disease")
