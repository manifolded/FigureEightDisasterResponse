import sys
import pandas as pd
import numpy as np
import sqlalchemy as sqal
from sklearn.model_selection import train_test_split
import spacy
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import pickle as pkl
import joblib

en_nlp = spacy.load('en')
stopwords = spacy.lang.en.stop_words.STOP_WORDS
stemmer = SnowballStemmer('english')


# genScoreTable()
# Takes the true and predicted y-values and returns a 2-D array containing the precision,
# recall and f1 scores for each target category.
def genScoreTable(true, pred, zero_division='warn'):
    table = np.empty(shape=(true.shape[1], 3))
    for i in range(true.shape[1]):
        table[i] = \
            [precision_score(true[:,i], pred[:,i], zero_division=zero_division),
            recall_score(true[:,i], pred[:,i], zero_division=zero_division),
            f1_score(true[:,i], pred[:,i], zero_division=zero_division)]
    return table

# printScores()
# Takes the true and predicted y-values, and their column labels, and generates a
# nicely formatted version of the genScoreTable() table.
def printScores(labels, true, pred, zero_division='warn'):
    results = genScoreTable(true, pred, zero_division=zero_division)
    print("{0:>20}".format(''), 'prec', 'recall', 'f1', sep='\t')
    for i in range(results.shape[0]):
        print("{0:>20}".format(labels[i]), '%.2f' % results[i, 0],
            '%.2f' % results[i,1], '%.2f' % results[i,2], sep = '\t')


# load_data()
# Takes the path to the database file containing the data and extracts the X and y
# data and the category labels and returns them.  Note that 'X' data is here called
# 'text' since the vectorizer isn't formally considered part of the pipeline.
def load_data(database_filepath):
    # open the database file created by previous script
    engine = sqal.create_engine('sqlite:///' + database_filepath)
    # and grab the table therein
    df = pd.read_sql_table('MessageCategorization', engine)

    in_columns = 'message'
    out_columns = list(df.columns)[4:]

    # remove outliers from 'related' column
    df['related'] = np.clip(df['related'], 0, 1)

    text = df[in_columns].values
    y = df[out_columns].values

    # # save some for data for testing the trained model
    # text_train, text_test, y_train, y_test = \
    #     train_test_split(text, y, test_size=0.33, random_state=42)

    return text, y, out_columns


# tokenize()
# Takes a string (the message) and normalizes it by first tokenizing and then
# lemmatizing the words using the spaCy library.  Finally the tokens are stemmed.
# The resulting list of tokens is returned.
#
# https://realpython.com/natural-language-processing-spacy-python/ was helpful in
# determining which among the bewildering array of options I should employ.
def tokenize(text):
    # tokenize the text using spacy's model for English
    doc = en_nlp(text)
    # while we lemmatize the now tokenized text, let's not forget to drop
    #   tokens that are stop_words or punctuation
    lemmas = [token.lemma_ for token in doc
        if token not in stopwords and not token.is_punct]
    # Had better luck with this nltk stemmer
    return [stemmer.stem(lemma) for lemma in lemmas]


# build_nlp_model()
# Construct the nlp first stage for the pipeline and return it
def build_nlp_model():
	return make_pipeline(
		TfidfVectorizer(tokenizer=tokenize, min_df=5))


# build_ml_model()
# Constructs the ml second stage for the pipeline and return it
def build_ml_model():
    return make_pipeline(
        MultiOutputClassifier(
            estimator=AdaBoostClassifier(
                base_estimator=DecisionTreeClassifier(max_depth=2),
                n_estimators=10, learning_rate=1)))


# build_model()
# Combine the two pieces and return the full pipeline
def build_model(nlp_model, ml_model):
	return make_pipeline(nlp_model, ml_model)


# evaluate_model()
# Takes the predicted y-values, y_predict, along with test samples, text_test,
# and true results, y_test, and uses them to output the test data cv score and
# the breakdown of precision, recall and f1 scores by target category.
def evaluate_model(model, text_test, y_test, y_predict, category_names):
	# y_pred = model.predict(text_test)
    print('Test data cv score = {0:.2f}'.format(model.score(text_test, y_test)))
    printScores(category_names, y_test, y_predict, zero_division=0)
    # return y_pred


# save_model()
# Saves the model to a pickle file
def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pkl.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        text, y, category_names = load_data(database_filepath)
        text_train, text_test, y_train, y_test = train_test_split(text, y, test_size=0.33)

        print('Building model...')
        nlp_model = build_nlp_model()
        ml_model = build_ml_model()
        model = build_model(nlp_model, ml_model)

        print('Training model...')
        X_train = nlp_model.fit_transform(text_train)
        ml_model.fit(X_train, y_train)
        X_test = nlp_model.transform(text_test)
        y_predict = ml_model.predict(X_test)

        print('Evaluating model...')
        evaluate_model(model, text_test, y_test, y_predict, category_names)

        print('Caching data...\n    FILE: predicted.joblib')
        with open('predicted.joblib', 'wb') as f:
            joblib.dump(y_predict, f)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        vocab = nlp_model['tfidfvectorizer'].vocabulary_
        with open('nlp_vocabulary.joblib', 'wb') as f:
            joblib.dump(vocab, f)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
