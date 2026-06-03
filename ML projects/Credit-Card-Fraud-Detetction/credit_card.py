import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================
# LOAD DATASET
# ==========================

train_df = pd.read_csv("fraudTrain.csv")
test_df = pd.read_csv("fraudTest.csv")

# Use a smaller sample for faster execution
train_df = train_df.sample(n=50000, random_state=42)
test_df = test_df.sample(n=10000, random_state=42)

print("Dataset Loaded Successfully!")

# ==========================
# DATA PREPROCESSING
# ==========================

drop_columns = [
    'Unnamed: 0',
    'trans_date_trans_time',
    'cc_num',
    'first',
    'last',
    'street',
    'city',
    'state',
    'zip',
    'dob',
    'trans_num'
]

train_df.drop(columns=drop_columns, inplace=True, errors='ignore')
test_df.drop(columns=drop_columns, inplace=True, errors='ignore')

# Encode categorical columns
categorical_cols = train_df.select_dtypes(include='object').columns

for col in categorical_cols:
    le = LabelEncoder()

    combined = pd.concat([
        train_df[col].astype(str),
        test_df[col].astype(str)
    ])

    le.fit(combined)

    train_df[col] = le.transform(train_df[col].astype(str))
    test_df[col] = le.transform(test_df[col].astype(str))

# ==========================
# FEATURES & TARGET
# ==========================

X_train = train_df.drop("is_fraud", axis=1)
y_train = train_df["is_fraud"]

X_test = test_df.drop("is_fraud", axis=1)
y_test = test_df["is_fraud"]

# ==========================
# MODEL TRAINING
# ==========================

print("Training Decision Tree Model...")

model = DecisionTreeClassifier(
    random_state=42,
    max_depth=10
)

model.fit(X_train, y_train)

print("Model Training Completed!")

# ==========================
# PREDICTIONS
# ==========================

y_pred = model.predict(X_test)

# ==========================
# EVALUATION
# ==========================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 50)
print("MODEL PERFORMANCE")
print("=" * 50)

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ==========================
# CONFUSION MATRIX
# ==========================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues'
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ==========================
# SAMPLE PREDICTIONS
# ==========================

print("\nSample Predictions:\n")

sample_results = pd.DataFrame({
    "Actual": y_test.values[:20],
    "Predicted": y_pred[:20]
})

print(sample_results)