from functions_preprocess import LinguisticPreprocessor, download_if_non_existent, fit_model
from datasets import load_dataset
import pickle
import string
from bs4 import BeautifulSoup
from textblob import TextBlob
import re
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download("punkt")
nltk.download('omw-1.4')
nltk.download("wordnet")
download_if_non_existent('corpora/stopwords', 'stopwords')
download_if_non_existent('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')
download_if_non_existent('corpora/wordnet', 'wordnet')


def main():
    #####load dataset
    data = load_dataset("rotten_tomatoes")
    com_data = load_dataset('sst2')
    imdb = load_dataset('imdb')
    X_train = data['train']['text']
    y_train = data['train']['label']
    X_test = data['test']['text']
    y_test = data['test']['label']
    X_train += com_data['train']['sentence']
    y_train += com_data['train']['label']
    X_train += imdb['train']['text']
    y_train += imdb['train']['label']
    X_test = np.array(X_test)
    X_train = np.array(X_train)

    pipeline = Pipeline(
    steps=[
        ("processor", LinguisticPreprocessor()),
        ("vectorizer", TfidfVectorizer(ngram_range=(1, 2))),
        ("model", SGDClassifier(loss="log_loss", n_jobs = -1, alpha=0.000001, penalty= 'elasticnet'))
    ]
    )

    fit_model(pipeline, X_train, y_train, X_test, y_test)

    preds = pipeline.predict(X_test)
    
    ##### prints the macro and weighted accuracies, and F-1 score
    print(classification_report(y_test, preds))
    
    ##### saving model 
    model_pkl_file = "sentiment_model.pkl"  

    with open(model_pkl_file, 'wb') as file:  
        pickle.dump(pipeline, file)


if __name__ == "__main__":
    main()
