from functions_preprocess import LinguisticPreprocessor, fit_model, training_data
from datasets import load_dataset
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


def main():
    #####load dataset
    dataset_1 = load_dataset("rotten_tomatoes")
    dataset_2 = load_dataset('sst2')
    dataset_2 = dataset_2.rename_column('sentence', 'text')
    dataset_3 = load_dataset('imdb')

    X_train, y_train, X_test, y_test = training_data(dataset_1, dataset_2, dataset_3)

    pipeline = Pipeline(
    steps=[
        ("processor", LinguisticPreprocessor()),
        ("vectorizer", TfidfVectorizer(ngram_range=(1, 2))),
        ("model", SGDClassifier(loss="log_loss", n_jobs = -1, alpha=0.000001, penalty= 'elasticnet'))])

    ####### fit model and save the results
    fit_model(pipeline, X_train, y_train, X_test, y_test)
    predictions = pipeline.predict(X_test)

    ####### Create a DataFrame with index and predictions
    results_df = pd.DataFrame({
    "index": range(len(predictions)),
    "pred": predictions})

    ####### Save the DataFrame to a CSV file
    results_df.to_csv("results.csv", index=False)


if __name__ == "__main__":
    main()


#    model_pkl_file = "sentiment_model.pkl"  
#
#    with open(model_pkl_file, 'wb') as file:  
#        pickle.dump(pipeline, file)
