import streamlit as st
from functions_preprocess import LinguisticPreprocessor, download_if_non_existent, CNN
import pickle
import nltk
from datasets import load_dataset
import torch
nltk.download('stopwords')
nltk.download('punkt')
download_if_non_existent('corpora/stopwords', 'stopwords')
download_if_non_existent('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')
download_if_non_existent('corpora/wordnet', 'wordnet')
from torchtext.data.utils import get_tokenizer


#################################################################### Streamlit interface
st.title("Movie Reviews: An NLP Sentiment analysis")

#################################################################### Cache the model loading

@st.cache_data()
def load_model():
  model_pkl_file = "sentiment_model.pkl"  
  with open(model_pkl_file, 'rb') as file:  
    model = pickle.load(file)
  return model

def load_cnn():
    model = CNN(16236, 300, 128, [3, 8], 0.5, 2)
    model.load_state_dict(torch.load('model_cnn.pkl', map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_sentiment(text, model, vocab, torch_text = False):
    tokenizer = get_tokenizer("basic_english")
    if torch_text == True:
        processor.transform(text)
        tokens = tokenizer(text)
        encoded = [vocab[token] for token in tokens]
        input_tensor = torch.tensor(encoded).unsqueeze(0).to(device)
        
        with torch.no_grad():  # No gradient needed
            model.eval()  # Evaluation mode
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
        
        return pred_class  # Return the predicted class index
    else:
        processor.transform(text)
        prediction = model.predict([text])
        return prediction


model_1 = load_model()
model_2 = load_cnn()
processor = LinguisticPreprocessor()
train_data = load_dataset('rotten_tomatoes', split='train')
vocab, tokenizer = build_vocab(train_data)


############################################################# Text input

with st.expander("Model 1: SGD Classifier"):
    st.markdown("Give it a go by writing a positive or negative text, and analyze it!")

    # Text input inside the expander
    user_input = st.text_area("Enter text here...", key='model1_input')
    if st.button('Analyze', key='model1_button'):
        # Displaying output
        result = predict_sentiment(user_input, model_1)
        if result >= 0.5:
            st.write('The sentiment is: Positive 😀', key='model1_poswrite')
        else:
            st.write('The sentiment is: Negative 😞', key='model1_negwrite')

with st.expander("Model 2: CNN Sentiment analysis"):
    st.markdown("Give it a go by writing a positive or negative text, and analyze it!")

    # Text input inside the expander
    user_input = st.text_area("Enter text here...", key='model2_input')
    if st.button('Analyze', key='model2_button'):
        # Displaying output
        result = predict_sentiment(user_input, model_2, vocab, torch_text=True)
        if result >= 0.5:
            st.write('The sentiment is: Positive 😀', key='model2_poswrite')
        else:
            st.write('The sentiment is: Negative 😞', key='model2_negwrite')

st.caption("Por @efeperro.")
stop_words = set(stopwords.words('english'))
