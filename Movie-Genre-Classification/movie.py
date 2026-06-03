import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# Load dataset
train_file = "C:\\Users\\ASUS\\Desktop\\ML projects\\Movie-Genre-Classification\\train_data.txt"
test_file = "C:\\Users\\ASUS\\Desktop\\ML projects\\Movie-Genre-Classification\\test_data.txt"
solution_file = "C:\\Users\\ASUS\\Desktop\\ML projects\\Movie-Genre-Classification\\test_data_solution.txt"

# Read training data
train_data = pd.read_csv(
    train_file,
    sep=" ::: ",
    names=["ID", "Genre", "Plot"],
    engine="python"
)

# Read test data
test_data = pd.read_csv(
    test_file,
    sep=" ::: ",
    names=["ID", "Plot"],
    engine="python"
)

# Read actual labels
solution_data = pd.read_csv(
    solution_file,
    sep=" ::: ",
    names=["ID", "Genre", "Plot"],
    engine="python"
)

# Features and labels
X_train = train_data["Plot"]
y_train = train_data["Genre"]

X_test = test_data["Plot"]
y_test = solution_data["Genre"]

# TF-IDF + Logistic Regression Pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english',
        max_features=10000
    )),
    ('classifier', LogisticRegression(
        max_iter=1000
    ))
])

# Train model
print("Training Model...")
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", round(accuracy * 100, 2), "%")
print("\nClassification Report:\n")
print(classification_report(y_test, predictions, zero_division=0))

# Custom Prediction
while True:
    plot = input("\nEnter movie plot summary (or type exit): ")

    if plot.lower() == "exit":
        break

    genre = model.predict([plot])[0]
    print("Predicted Genre:", genre)