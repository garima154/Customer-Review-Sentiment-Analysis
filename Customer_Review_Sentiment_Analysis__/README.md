# 🎬 Customer Review Sentiment Analysis using NLP

## 📌 Project Overview

This project is a Natural Language Processing (NLP) application that classifies customer/movie reviews as **Positive** or **Negative** using Machine Learning.

The project uses **TF-IDF Vectorization** for feature extraction and **Logistic Regression** for sentiment classification. It also includes data visualization techniques such as sentiment distribution, confusion matrix, and word clouds to better understand the dataset.

---

## 🚀 Features

- Load and preprocess customer reviews
- Clean and normalize text data
- Convert text into numerical features using TF-IDF
- Train a Logistic Regression model
- Evaluate model performance
- Generate Confusion Matrix
- Visualize Sentiment Distribution
- Generate Positive and Negative Word Clouds
- Save trained model and vectorizer
- Predict sentiment for custom user reviews

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Matplotlib
- Seaborn
- WordCloud
- Joblib

---

## 📂 Project Structure

```
Customer_Review_Sentiment_Analysis/

│── IMDB Dataset.csv
│── review_dataset.csv
│── train_model.py
│── phase8.py
│── phase8_predict.py
│── sentiment_model.pkl
│── tfidf_vectorizer.pkl
│── requirements.txt
│── README.md
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/your-username/Customer_Review_Sentiment_Analysis.git
```

Navigate to the project directory

```bash
cd Customer_Review_Sentiment_Analysis
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

### Train the Model

```bash
python train_model.py
```

This will generate:

- sentiment_model.pkl
- tfidf_vectorizer.pkl

---

### Run Prediction

```bash
python phase8_predict.py
```

Example:

```
Enter a review:

This movie was absolutely amazing!

Prediction:
Positive 😊
```

---

## 📊 Model Details

**Machine Learning Algorithm**

- Logistic Regression

**Feature Extraction**

- TF-IDF Vectorizer
- N-grams (Unigrams + Bigrams)

---

## 📈 Visualizations

The project generates:

- Sentiment Distribution
- Confusion Matrix
- Positive Review Word Cloud
- Negative Review Word Cloud

---

## 📷 Sample Output

```
Enter a review:

Worst movie ever.

Prediction:
Negative 😞
```

```
Enter a review:

Excellent acting and amazing story.

Prediction:
Positive 😊
```

---

## 📚 Dataset

Dataset Used:

**IMDb Movie Reviews Dataset**

The dataset contains thousands of labeled movie reviews with positive and negative sentiments.

---

## 📌 Future Improvements

- Support Neutral Sentiment
- Deploy using Streamlit
- Build a Flask Web Application
- Deep Learning using LSTM
- BERT-based Sentiment Analysis
- Real-time sentiment prediction from user input

---

## 👩‍💻 Author

**Garima Pandey**

Artificial Intelligence & Machine Learning Student

GitHub: https://github.com/your-github-username

LinkedIn: https://linkedin.com/in/your-linkedin-profile

---

## ⭐ If you found this project useful, consider giving it a Star on GitHub!