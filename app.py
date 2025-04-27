import streamlit as st
import joblib

# Load vectorizer and model
vectorizer = joblib.load("vectorizer.pkl")  # Save TF-IDF vectorizer
model = joblib.load("model.pkl")            # Save trained model

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK resources
nltk.download('stopwords', quiet = True)
nltk.download('punkt', quiet = True)

stop_words = set(stopwords.words('english'))

# Clean text function
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#','', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    filtered = [w for w in tokens if not w in stop_words]
    return ' '.join(filtered)

# Streamlit UI
st.title("Tweet Sentiment Analyzer")
st.write("Enter a tweet below to predict its sentiment.")

user_input = st.text_area("Enter tweet here:")

if st.button("Analyze"):
    cleaned = clean_text(user_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    
    st.write("### Prediction:")
    st.success(prediction.capitalize())