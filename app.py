import streamlit as st
import pandas as pd
import re
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

# Define constants (should match those used in training)
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 100

@st.cache_data
def load_data(filepath="Final.csv"):
    """Loads the dataset from a CSV file."""
    try:
        df = pd.read_csv(filepath, engine='python')
        return df
    except pd.errors.ParserError as e:
        st.error(f"Error loading CSV file: {e}")
        return None

@st.cache_data
def preprocess_text(df):
    """Combines summary and reviewText and cleans the text."""
    if df is not None:
        df['text'] = df['summary'].fillna('') + " " + df['reviewText'].fillna('')

        def clean_text(text):
            text = text.lower()
            text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
            return text

        df['text'] = df['text'].apply(clean_text)
        return df
    return None

@st.cache_resource
def load_tokenizer(filepath="tokenizer.pickle"):
    """Loads the saved tokenizer."""
    try:
        with open(filepath, "rb") as handle:
            tokenizer = pickle.load(handle)
        return tokenizer
    except FileNotFoundError:
        st.error(f"Tokenizer file not found at {filepath}")
        return None

@st.cache_resource
def load_model(filepath="sentiment_model.h5"):
    """Loads the trained sentiment analysis model."""
    try:
        model = tf.keras.models.load_model(filepath)
        return model
    except FileNotFoundError:
        st.error(f"Model file not found at {filepath}")
        return None

@st.cache_resource
def load_spacy_model(model_name="en_core_web_sm"):
    """Loads the spaCy model for NER."""
    try:
        nlp = spacy.load(model_name)
        return nlp
    except:
        st.error(f"SpaCy model '{model_name}' not found. Please download it using '!python -m spacy download {model_name}' in your environment.")
        return None


def predict_sentiment(text, tokenizer, model, max_sequence_length=MAX_SEQUENCE_LENGTH):
    """Predicts the sentiment of a given text."""
    if not tokenizer or not model:
        return "Error: Model or tokenizer not loaded."

    # Preprocess text
    cleaned_text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length, padding='post')

    # Predict
    prediction = model.predict(padded_sequence)
    sentiment_label = tf.argmax(prediction, axis=1).numpy()[0]

    # Map label to sentiment
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_map.get(sentiment_label, "Unknown")

def generate_wordcloud(text_data):
    """Generates and displays a word cloud."""
    if not text_data.empty:
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(" ".join(text_data.dropna().astype(str)))
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    else:
        st.warning("No text data available for word cloud.")


def display_named_entities(text_data, nlp):
    """Displays named entities from the text data."""
    if not nlp:
        st.warning("SpaCy model not loaded.")
        return
    if not text_data.empty:
        doc = nlp(" ".join(text_data.dropna().astype(str)[:100])) # Process a subset for performance

        st.subheader("Named Entities:")
        if doc.ents:
            for ent in doc.ents:
                st.write(f"- {ent.text} ({ent.label_})")
        else:
            st.write("No named entities found in the provided text.")
    else:
         st.warning("No text data available for Named Entity Recognition.")


def display_most_common_words(text_data, n=20):
    """Displays the most common words."""
    if not text_data.empty:
        words = re.findall(r'\b\w+\b', " ".join(text_data.dropna().astype(str)).lower())
        word_counts = Counter(words)
        st.subheader(f"Most Common {n} Words:")
        if word_counts:
            for word, count in word_counts.most_common(n):
                st.write(f"- {word}: {count}")
        else:
            st.write("No words found in the provided text.")
    else:
        st.warning("No text data available for word counts.")


# Streamlit App
st.title("Sentiment Analysis of Book Reviews")

st.write("Enter a book review to analyze its sentiment (Negative, Neutral, or Positive).")

# Load data, tokenizer, model, and spaCy model
df = load_data()
df = preprocess_text(df) # Preprocess the loaded data
tokenizer = load_tokenizer()
model = load_model()
nlp = load_spacy_model()

# Text input for sentiment analysis
user_input = st.text_area("Enter your review here:")

if st.button("Analyze Sentiment"):
    if user_input and tokenizer and model:
        sentiment = predict_sentiment(user_input, tokenizer, model)
        st.subheader("Predicted Sentiment:")
        st.write(sentiment)
    elif not user_input:
        st.warning("Please enter some text to analyze.")

st.sidebar.subheader("Visualizations from Dataset")

if df is not None:
    if st.sidebar.checkbox("Show Word Cloud"):
        st.subheader("Word Cloud of Reviews")
        generate_wordcloud(df['reviewText'])

    if st.sidebar.checkbox("Show Named Entities"):
        if nlp:
            st.subheader("Named Entities in Reviews")
            display_named_entities(df['reviewText'], nlp)
        else:
            st.warning("SpaCy model not loaded, cannot display named entities.")

    if st.sidebar.checkbox("Show Most Common Words"):
        st.subheader("Most Common Words in Reviews")
        display_most_common_words(df['reviewText'])
else:
    st.warning("Dataset not loaded, visualizations are not available.")
