import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ==============================
# 1. Load Dataset
# ==============================
data = pd.read_csv("student_question_answer_dataset_1000.csv")
data.columns = ["question", "answer"]

print("Total samples:", len(data))

# ==============================
# 2. Features and Labels
# ==============================
X = data["question"]
y = data["answer"]

# ==============================
# 3. Train-Test Split (80/20)
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# ==============================
# 4. Dataset Split Graph
# ==============================
plt.figure()
plt.bar(["Training Data", "Testing Data"], [len(X_train), len(X_test)])
plt.ylabel("Number of Questions")
plt.title("Dataset Split (80% Train / 20% Test)")
plt.show()

# ==============================
# 5. TF-IDF Vectorization
# ==============================
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ==============================
# 6. Train ML Model
# ==============================
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ==============================
# 7. Model Evaluation
# ==============================
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

# ==============================
# 8. Accuracy Graph
# ==============================
plt.figure()
plt.bar(["Accuracy"], [accuracy * 100])
plt.ylim(0, 100)
plt.ylabel("Percentage")
plt.title("Model Accuracy")
plt.show()

# ==============================
# 9. Save Model and Vectorizer
# ==============================
joblib.dump(model, "student_chatbot_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully")

# ==============================
# 10. Chatbot with Confidence Check
# ==============================
CONFIDENCE_THRESHOLD = 0.45

def student_doubt_solver(user_question):
    question_vec = vectorizer.transform([user_question])
    probabilities = model.predict_proba(question_vec)[0]
    max_prob = np.max(probabilities)

    if max_prob < CONFIDENCE_THRESHOLD:
        return "Sorry, I cannot help with this question."
    else:
        return model.classes_[np.argmax(probabilities)]

# ==============================
# 11. User Interaction
# ==============================
print("\nStudent Doubt Solver Chatbot")
print("Type 'exit' to stop")

while True:
    user_input = input("\nAsk your question: ")
    if user_input.lower() == "exit":
        break
    print("Answer:", student_doubt_solver(user_input))
