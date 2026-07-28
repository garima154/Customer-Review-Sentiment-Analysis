import joblib

# Load saved model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

while True:
    review = input("\nEnter a review (or type 'exit' to quit): ")

    if review.lower() == "exit":
        break

    review_vector = vectorizer.transform([review])

    prediction = model.predict(review_vector)

    if prediction[0] == "positive":
        print("Prediction: Positive 😊")
    else:
        print("Prediction: Negative 😞")