import streamlit as st 
from functions_preprocess import LinguisticPreprocessor, download_if_non_existent, fit_model
import pickle
import string
from bs4 import BeautifulSoup
from textblob import TextBlob
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download("punkt")
nltk.download('omw-1.4')
nltk.download("wordnet")
download_if_non_existent('corpora/stopwords', 'stopwords')
download_if_non_existent('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')
download_if_non_existent('corpora/wordnet', 'wordnet')




#################################################################### Streamlit interface
st.title("Movie Reviews: An NLP Sentiment analysis")

st.markdown("### NLP Processing utilizing various ML approaches")
st.markdown("This initial approach merges multiple datasets to predict a positive or negative sentiment of reviews.")


#################################################################### Cache the model loading
@st.cache_data()
def load_model():
    model_pkl_file = "sentiment_model.pkl"  
    with open(model_pkl_file, 'rb') as file:  
        model = pickle.load(file)
    return model

model = load_model()
processor = LinguisticPreprocessor()
def predict_sentiment(text, model):
    processor.transform(text)
    prediction = model.predict([text])
    return prediction


############################################################# Text input
user_input = st.text_area("Enter text here...")

if st.button('Analyze'):
    # Displaying output
    result = predict_sentiment(user_input, model)
    if result >= 0.5:
        st.write('The sentiment is: Positive 😀')
    else:
        st.write('The sentiment is: Negative 😞')


st.caption("Made by @efeperro con mucho ❤️. Credits to 🤗")


############################################################### some global variables required
lemmatizer = WordNetLemmatizer()
tokenizer = Tokenizer()
stop_words = set(stopwords.words('english'))