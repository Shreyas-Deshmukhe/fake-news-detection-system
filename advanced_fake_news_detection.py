import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load datasets
fake = pd.read_csv(r"Fake_folder\Fake.csv")
true = pd.read_csv(r"True_folder\True.csv")

# Add labels
fake["label"] = "FAKE"
true["label"] = "REAL"

# Combine datasets
data = pd.concat([fake, true], ignore_index=True)

# Keep only text and label columns
data = data[["text", "label"]]

# Features and target
X = data["text"]
y = data["label"]

# Convert text into numerical form
vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(X)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

# User input
news = input("\nEnter a news statement: ")

# Convert user input to vector form
news_vector = vectorizer.transform([news])

# Predict
prediction = model.predict(news_vector)

print("Prediction:", prediction[0])