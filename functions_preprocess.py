
import string
import pandas as pd
from bs4 import BeautifulSoup
import re
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.metrics import ConfusionMatrixDisplay
from keras.preprocessing.text import Tokenizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


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

def training_data(dataset_1, dataset_2, dataset_3):
  X_test = dataset_1['test']['text']
  y_test = dataset_1['test']['label']

  test_df = pd.DataFrame({
      'text':X_test,
      'label': y_test
  })

  combined_train_df = pd.DataFrame({
      'text': dataset_1['train']['text'] + dataset_2['train']['text'] + dataset_3['train']['text'],
      'label': dataset_1['train']['label'] + dataset_2['train']['label'] + dataset_3['train']['label']
  })

  combined_train_df.drop_duplicates(subset=['text'], inplace=True)

  merged_df = pd.merge(combined_train_df, test_df, on="text", how='left', indicator=True)
  result_df = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])


  X_train = result_df['text'].tolist()
  y_train = result_df['label_x'].tolist()
  X_test = np.array(X_test)
  X_train = np.array(X_train)

  return X_train, y_train, X_test, y_test

  def _lemmatize_one_sentence(self, sentence):
    sentence = nltk.word_tokenize(sentence)
    sentence = list(map(lambda word: self.lemmatizer.lemmatize(word), sentence))
    return " ".join(sentence)
