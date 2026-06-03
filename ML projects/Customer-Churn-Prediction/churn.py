# ============================================
# CUSTOMER CHURN PREDICTION USING MACHINE LEARNING
# ============================================

# Import Libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import accuracy_score


# ============================================
# LOAD DATASET
# ============================================

data = pd.read_csv(r"C:\Users\ASUS\Desktop\ML projects\Customer-Churn-Prediction\Churn_Modelling.csv")

print("\nDataset Loaded Successfully")


# ============================================
# REMOVE UNNECESSARY COLUMNS
# ============================================

data.drop(
    ["RowNumber", "CustomerId", "Surname"],
    axis=1,
    inplace=True
)


# ============================================
# ENCODE CATEGORICAL DATA
# ============================================

gender_encoder = LabelEncoder()
geo_encoder = LabelEncoder()

data["Gender"] = gender_encoder.fit_transform(data["Gender"])

data["Geography"] = geo_encoder.fit_transform(data["Geography"])


# ============================================
# DEFINE FEATURES AND TARGET
# ============================================

X = data.drop("Exited", axis=1).values

y = data["Exited"].values


# ============================================
# SPLIT DATA
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================
# FEATURE SCALING
# ============================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)


# ============================================
# LOGISTIC REGRESSION MODEL
# ============================================

log_model = LogisticRegression(max_iter=1000)

log_model.fit(X_train, y_train)

log_predictions = log_model.predict(X_test)

log_accuracy = accuracy_score(y_test, log_predictions)


# ============================================
# RANDOM FOREST MODEL
# ============================================

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)


# ============================================
# GRADIENT BOOSTING MODEL
# ============================================

gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)

gb_model.fit(X_train, y_train)

gb_predictions = gb_model.predict(X_test)

gb_accuracy = accuracy_score(y_test, gb_predictions)


# ============================================
# DISPLAY MODEL ACCURACY
# ============================================

print("\n========== MODEL ACCURACY COMPARISON ==========\n")

print("Logistic Regression Accuracy :",
      round(log_accuracy * 100, 2), "%")

print("Random Forest Accuracy :",
      round(rf_accuracy * 100, 2), "%")

print("Gradient Boosting Accuracy :",
      round(gb_accuracy * 100, 2), "%")


# ============================================
# BEST MODEL
# ============================================

best_accuracy = max(
    log_accuracy,
    rf_accuracy,
    gb_accuracy
)

if best_accuracy == log_accuracy:
    best_model = "Logistic Regression"

elif best_accuracy == rf_accuracy:
    best_model = "Random Forest"

else:
    best_model = "Gradient Boosting"

print("\nBest Performing Model :", best_model)


# ============================================
# USER INPUT SECTION
# ============================================

print("\n========== ENTER CUSTOMER DETAILS ==========\n")

credit_score = int(input("Enter Credit Score: "))

geography = int(input(
    "Enter Geography (0-France, 1-Germany, 2-Spain): "
))

gender = int(input(
    "Enter Gender (0-Female, 1-Male): "
))

age = int(input("Enter Age: "))

tenure = int(input("Enter Tenure: "))

balance = float(input("Enter Balance: "))

products = int(input("Enter Number of Products: "))

credit_card = int(input(
    "Has Credit Card? (1-Yes, 0-No): "
))

active_member = int(input(
    "Is Active Member? (1-Yes, 0-No): "
))

salary = float(input("Enter Estimated Salary: "))


# ============================================
# CREATE CUSTOMER DATA
# ============================================

sample_customer = [[
    credit_score,
    geography,
    gender,
    age,
    tenure,
    balance,
    products,
    credit_card,
    active_member,
    salary
]]


# ============================================
# SCALE CUSTOMER DATA
# ============================================

sample_customer = scaler.transform(sample_customer)


# ============================================
# PREDICT CUSTOMER CHURN
# ============================================

prediction = gb_model.predict(sample_customer)

probability = gb_model.predict_proba(sample_customer)


# ============================================
# FINAL RESULT
# ============================================

print("\n========== CUSTOMER CHURN PREDICTION ==========\n")

if prediction[0] == 1:
    print("Prediction Result : Customer Will Leave")

else:
    print("Prediction Result : Customer Will Stay")

print("\nChurn Probability :",
      round(probability[0][1] * 100, 2), "%")