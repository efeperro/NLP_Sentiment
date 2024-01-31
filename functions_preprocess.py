
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


def download_if_non_existent(res_path, res_name):
  try:
    nltk.data.find(res_path)
  except LookupError:
    print(f'resource {res_path} not found. Downloading now...')
    nltk.download(res_name)

def fit_model(pipeline, x_train, y_train, x_test, y_test):
  pipeline.fit(x_train, y_train)
  return ConfusionMatrixDisplay.from_estimator(pipeline, x_test, y_test, normalize="true")

class LinguisticPreprocessor(TransformerMixin):
  def __init__(self, ):
    super().__init__()
    self.lemmatizer = WordNetLemmatizer()
    self.tokenizer = Tokenizer()
    self.stop_words = set(stopwords.words('english'))
    self.stop = stopwords.words('english')

  def fit(self, X, y=None):
    return self

  def transform(self, X, y=None):
    X = self._remove_html_tags(X)
    X = self._remove_all_punctuations(X)
    X = self._remove_double_spaces(X)
    X = self._lemmatize(X)
    X = self._remove_stopwords(X)
    return X

  def _remove_html_tags(self, X):
    X = list(map( lambda x: BeautifulSoup(x, 'html.parser').get_text(), X))
    return X

  def _remove_all_punctuations(self, X):
    X = list(
        map(
            lambda text: re.sub('[%s]' % re.escape(string.punctuation), '', text),
            X
        )
    )
    return X

  def _remove_double_spaces(self, X):
    X = list(map(lambda text: re.sub(" +", " ", text), X))
    return X

  def _remove_stopwords(self, X):
    X = list(map(
            lambda text:  " ".join(
                [
                    word for word in text.split() if word not in (self.stop_words)
                ]
            ),
            X
        )
    )
    return X

  def _lemmatize(self, X):
    X = list(map(lambda text: self._lemmatize_one_sentence(text), X))
    return X

  def _lemmatize_one_sentence(self, sentence):
    sentence = nltk.word_tokenize(sentence)
    sentence = list(map(lambda word: self.lemmatizer.lemmatize(word), sentence))
    return " ".join(sentence)
