import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# -------------------------------
# Load Dataset
# -------------------------------

df = pd.read_csv("IMDB Dataset.csv")

# -------------------------------
# NLP Initialization
# -------------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -------------------------------
# Text Cleaning
# -------------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(r'<.*?>', '', text)

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [word for word in words if word not in stop_words]

    words = [lemmatizer.lemmatize(word) for word in words]

    return " ".join(words)

df["clean_review"] = df["review"].apply(clean_text)

# -------------------------------
# Convert Labels
# -------------------------------

df["sentiment"] = df["sentiment"].map({
    "positive": 1,
    "negative": 0
})

# -------------------------------
# TF-IDF
# -------------------------------

tfidf = TfidfVectorizer(max_features=5000)

X = tfidf.fit_transform(df["clean_review"])

y = df["sentiment"]

# -------------------------------
# Train-Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# -------------------------------
# Train Model
# -------------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# -------------------------------
# Predictions
# -------------------------------

y_pred = model.predict(X_test)

# -------------------------------
# Visualization 1
# Sentiment Distribution
# -------------------------------

plt.figure(figsize=(6,5))

sns.countplot(x=df["sentiment"])

plt.title("Sentiment Distribution")

plt.xlabel("Sentiment")

plt.ylabel("Count")

plt.show()

# -------------------------------
# Visualization 2
# Confusion Matrix
# -------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative","Positive"],
    yticklabels=["Negative","Positive"]
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()

# -------------------------------
# Visualization 3
# Positive Word Cloud
# -------------------------------

positive_text = " ".join(
    df[df["sentiment"]==1]["clean_review"]
)

wordcloud = WordCloud(
    width=900,
    height=500,
    background_color="white"
).generate(positive_text)

plt.figure(figsize=(12,6))

plt.imshow(wordcloud)

plt.axis("off")

plt.title("Positive Reviews Word Cloud")

plt.show()

# -------------------------------
# Visualization 4
# Negative Word Cloud
# -------------------------------

negative_text = " ".join(
    df[df["sentiment"]==0]["clean_review"]
)

wordcloud = WordCloud(
    width=900,
    height=500,
    background_color="black"
).generate(negative_text)

plt.figure(figsize=(12,6))

plt.imshow(wordcloud)

plt.axis("off")

plt.title("Negative Reviews Word Cloud")

plt.show()