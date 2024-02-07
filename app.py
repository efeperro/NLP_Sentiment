import streamlit as st 
from functions_preprocess import LinguisticPreprocessor
import pickle


#################################################################### Streamlit interface
st.title("Movie Reviews: An NLP Sentiment analysis")

st.markdown("### NLP Processing utilizing various ML approaches")
st.markdown("##### This initial approach merges multiple datasets, processed through a TF-IDF vectorizer with 2 n-grams and fed into a Stochastic Gradient Descent model.")
st.markdown("Give it a go by writing a positive or negative text, and analyze it!")


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


st.caption("Por @efeperro con ❤️. Credits to 🤗")

tokenizer = Tokenizer()
stop_words = set(stopwords.words('english'))
