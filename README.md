
# NLP: A Movie Sentiment Analysis

## Description

This project emphasizes the approaches of natural language processing (NLP), through different methods towards model building. In this specific case, the approach is throguh classical machine learning models. The main script (`main.py`) orchestrates the workflow, including data preprocessing, model training, and saving predictions for model accuracy test. 

The datasets included are 'rotten-tomatoes', 'SST-2', and 'imdb'; these are all loaded from HuggingFace Datasets where 'sst-2' and 'imdb' function as complementary support for diverse observations. Although there are many datasets used, the test set will only be on 'rotten-tomatoes' dataset.
The preprocessing tasks Regarding the trials for the final model, the baseline model was a logistic regression which yielded a 78% macro accuracy score. Many other approaches were attempted with very little improvement towards the accuracy, which will be listed below in the Modeling.

If you would like to interact with the final model's streamlit interface, it is deployed in HuggingFace Spaces through [this link](https://huggingface.co/spaces/efeperro/Movie_Analyzer)

## Getting Started

### Dependencies

The dependencies present in the project are listed in the `requirements.txt` which are essential for `main.py` to function:

* `Datasets` for accessing our primary text datasets.
* `nltk`, `keras`, `BeautifulSoup` and `textblob` for NLP.
* `pandas` and `numpy` for data wrangling and structurization.
* `scikit-learn` for model building.

### Installing

- The files must be in the same directory in order for the reference of `functions_preprocess.py` to properly work, and this way the file export of results.csv can be saved somewhere desired.

### Executing program

- How to run the program
* Download `requirements.txt`, `main.py`, and `functions_preprocess.py`.
* Direct to your terminal to install the requirements and then run `main.py` as specified below. Make sure the `functions_preprocess.py` file is in the same directory as `main.py`.

* For Unix/Linux/Mac
```bash
echo "Installing dependencies..."
pip install -r requirements.txt
echo "Running main script..."
python main.py
```

* For Windows
```batch
@echo off
echo Installing dependencies...
pip install -r requirements.txt
echo Running main script...
python main.py
```


## Preprocessing

### Data Loading and Wrangling
The preprocessing steps are defined in `functions_preprocess.py` and include:

- Description of preprocessing steps (Placeholder for actual steps).
* After loading the data, the 'sst-2' text column is set to match the 2 other datasets.
* Duplicates were efficiently removed from the train data after merging all train data from the three datasets, later the train data was processed to remove any text case observed in the test data from the train data.
* 

 We meticulously removed duplicates to ensure the model's generalizability and fairness in evaluation.

### Pipeline

After splitting the data intro train and test sets and classify it as "Free from Data Leak", we head onto the pipeline text transformation

* Text processor removes HTML tags with `BeautifulSoup`.
* Double spaces and punctuations removel using `re`.
* Sentence lemmatization and removal of stopwords with `nltk`.

### Vectorizer

The TF-IDF vectorizer performed best out of all trials (including Hash and Count vectorizers), and allows the process to vectorization for the proper model representation.

## Modeling

The model training and prediction tasks are conducted in `main.py`, involving:

- Model used: *SGDClassifier*
- The SGD classifier proved to have the best score among all of the models without any additional optimization. Although Naive Bayes model showed similar scores, the SGD classifier was chosen for hyper-parameter tuning.

Other used models:
- Logistic Regression
- Naive Bayes
- XGBoost
- Random Forest Classifier
- Support Vector Machines

A cross-validation grid search was performed to take into acocunt the best combination of parameters for the model, among all parameters these were the ones tested:

```
search parameters = {
        "vectorizer__max_df": (0.5, 0.75, 1.0),
        "vectorizer__max_features": (None, 5000, 10000, 50000),
        "vectorizer__ngram_range": ((1, 1), (1, 2), (2, 2)),
        "model__alpha": (0.00001, 0.000001),
        "model__penalty": ("l2", "elasticnet")
    }
```

## Results

Results and evaluation metrics of the model mainly highlight:

- The hyper-parameter tuning properly works on the training with the following parameters: `loss="log_loss", n_jobs = -1, alpha=0.000001, penalty= 'elasticnet'`

- The model's final score form the 'rotten-tomatoes' dataset predictions is a 92% of proper positive review classifications, and 85% of proper negative classifications.
- A 0.88 score on both the accuracy, and macro average score.

Although these scores seem to good to be true, similar metrics were encountered with both test sets from the 'sst-2' and 'imdb' datasets.

## Authors

#### Fabian Perez


```bash
git clone https://yourprojectlink.git
cd yourprojectname
pip install -r requirements.txt
