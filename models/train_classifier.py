import sys
import pandas as pd
import numpy as np
import sqlalchemy as sqal
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import pickle as pkl
import joblib


def gen_token_table(model, n_vocab, n_categories):
    """Computes a prediction for every token in the vocabulary, generating
    what I call a 'token table'.  Used to associate output features with
    tokens.

    INPUT:
        model - (estimator or pipeline) The estimator to be used to make
            predictions.  Note:  This is probably *not* your full pipeline with
            vectorizer, but just the classifier back half.
        n_vocab - (int) Number of tokens in vocabulary
        n_categories - (int) Number of output features
    OUTPUT:
        result - (numpy.array 2D) Computed token table
    """
    result = np.zeros(shape = (n_vocab, n_categories))
    for i in range(n_vocab):
        # construct token vector for single token
        token_vector = np.zeros(shape=n_vocab)
        token_vector[i] = 1
        # compute categories for that single token and append to table
        result[i] = model.predict([token_vector])

    return result


def gen_canon_table(token_table, vocab):
    """Generates a 'canon table' with token vectors replaced with actual
    readable tokens.

    INPUT:
        token_table - (numpy.array 2D) binary table shape=(n_vocab, n_features)
        vocab - (numpy.array) maps token vectors to tokens
    OUTPUT:
        canon_table - (list of lists) each list contains tokens associated
            with a particular classifier feature
    """
    result = []
    for i in range(token_table.shape[1]):
        category_vector = token_table[:,i]
        token_indices = np.where(category_vector == 1)[0]
        # A wily trick for indexing a list
        result.append(list(np.array(vocab)[token_indices]))

    return(result)


def gen_score_table(true, pred, zero_division='warn'):
    """Generates a 2-D array containing the precision, recall and f1 scores for
    each target category.

    INPUT:
        true - (pandas.DataFrame) actual y-values from test set
        pred - (pandas.DataFrame) predicted y-values for test set
        zero_division - argument passed to scoring functions dictating how
            divide-by-zero errors are to be handled
    OUTPUT:
        table - (numpy.array) Array containing all the scores
    """
    table = np.empty(shape=(true.shape[1], 3))
    for i in range(true.shape[1]):
        table[i] = \
            [precision_score(true[:,i], pred[:,i],
                zero_division=zero_division),
            recall_score(true[:,i], pred[:,i], zero_division=zero_division),
            f1_score(true[:,i], pred[:,i], zero_division=zero_division)]

    return table


def print_scores(labels, true, pred, zero_division='warn'):
    """Generates a nicely formatted version of the gen_score_table() table.

    INPUT:
        labels - (list) Output feature (or category) names
        true - (pandas.DataFrame) actual y-values from test set
        pred - (pandas.DataFrame) predicted y-values for test set
        zero_division - argument passed to scoring functions dictating how
            divide-by-zero errors are to be handled
    """
    results = gen_score_table(true, pred, zero_division=zero_division)
    print("{0:>20}".format(''), 'prec', 'recall', 'f1', sep='\t')
    for i in range(results.shape[0]):
        print("{0:>20}".format(labels[i]), '%.2f' % results[i, 0],
            '%.2f' % results[i,1], '%.2f' % results[i,2], sep = '\t')


def grid_search(model, text_train, y_train):
    """Employs GridSearchCV to tune various parameters for the pipeline.
    At the moment, we're tuning the C parameter of LinearSVC.  Returns
    the optimized classifier.

    INPUT:
        model - (estimator or pipeline) pre-optimization model
        text_train - (numpy.array) training set sample messages
        y_train - (numpy.array) training set target values
    OUTPUT:
        grid - (sklearn.model_selection._search.GridSearchCV) meta_estimator
            optimized during grid search
    """
    # perform search over the C parameter of LinearSVC
    param_grid = {'multioutputclassifier__estimator__C': [0.03, 0.1, 0.3, 1.0]}
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
    grid.fit(text_train, y_train)
    return grid


def load_data(database_filepath):
    """Fetches dataset back from database file and extracts X and y data.
    Note that 'X' data is here called 'text'.

    INPUT:
        database_filepath - (str) path to the database file
    OUTPUT:
        text - (numpy.array) messages in strs
        y - (numpy.array 2D) target feature values for dataset
        out_columns - names of each of the target features
    """
    # open the database file created by previous script
    engine = sqal.create_engine('sqlite:///' + database_filepath)
    # and grab the table therein
    df = pd.read_sql_table('MessageCategorization', engine)

    # drop all negative category to allow use of LinearSVC
    df.drop('child_alone', axis=1, inplace=True)

    in_columns = 'message'
    out_columns = list(df.columns)[4:]

    text = df[in_columns].values
    y = df[out_columns].values

    # # save some for data for testing the trained model
    # text_train, text_test, y_train, y_test = \
    #     train_test_split(text, y, test_size=0.33, random_state=42)

    return text, y, out_columns


def tokenize(text):
    """Normalizes text in input message 'text' by lemmatizing words into
    tokens. Also removes URLs and stopwords.

    INPUT:
        text - (str) input message to be tokenized
    OUTPUT:
        tokens - (list) of tokens
    """
    # This experiment convinced me to lemmatize only rather than lemmatize and
    # stem.  I also got this nifty URL detector there.
    #https://gist.github.com/rajatsharma369007/de1e2024707ad90a73226643c314b118

    # initialization
    lemmatizer = WordNetLemmatizer()
    stop = stopwords.words("english")

    # Replaced all URLs with 'urlplaceholder'
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|'+\
                '(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    for url in re.findall(url_regex, text):
        text = text.replace(url, "urlplaceholder")

    # tokenize and lemmatize
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token).lower().strip() for
              token in tokens if token not in stop]

    return tokens


def build_model():
    """Construct the full pipeline.

    OUTPUT:
        model - (pipeline) model object
    """
    return make_pipeline(
        CountVectorizer(tokenizer=tokenize, min_df=10, max_df=0.10),
        TfidfTransformer(use_idf=True, smooth_idf=True, sublinear_tf=True),
        MultiOutputClassifier(estimator=
            LinearSVC(C=0.2, dual=False, multi_class='ovr',
                fit_intercept=True, max_iter=100)))


def evaluate_model(model, text_test, y_test, y_predict, category_names):
    """Prints the test set cv score and the breakdown of precision,
    recall and f1 scores by target feature.

    INPUT:
        model - (estimator or pipeline) our full classifier pipeline
        text_test - (numpy.array) test set sample messages
        y_test - (numpy.array) test set target feature values
        y_predict - (numpy.array) predicted target feature values for test set
        category_names - (list) output feature names
    """
    print('Test data cv score = {0:.2f}'.format(model.score(
        text_test, y_test)))
    print_scores(category_names, y_test, y_predict, zero_division=0)


def save_model(model, model_filepath):
    """Saves the model to a pickle file.

    INPUT:
        model - (estimator or pipeline) model to be saved
        model_filepath - (str) save to this pickle file
    """
    with open(model_filepath, 'wb') as f:
        pkl.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        text, y, category_names = load_data(database_filepath)
        text_train, text_test, y_train, y_test \
            = train_test_split(text, y, test_size=0.33)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(text_train, y_train)

        print('Tuning parameters...')
        grid = grid_search(model, text_train, y_train)
        y_predict = grid.predict(text_test)

        print('Evaluating model...')
        model = grid.best_estimator_
        evaluate_model(model, text_test, y_test, y_predict,
                       category_names)
#        vocab = nlp_model['tfidfvectorizer'].vocabulary_
        vocab = list(model['countvectorizer'].get_feature_names())
        n_voc = len(vocab)
        tail_model = make_pipeline(model['tfidftransformer'],
                                   model['multioutputclassifier'])
        token_table = gen_token_table(tail_model, n_voc, y_test.shape[1])
        canon_table = gen_canon_table(token_table, vocab)

        print('Caching data...\n    FILE: predicted.joblib')
        with open('predicted.joblib', 'wb') as f:
            joblib.dump(y_predict, f)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)
        with open('nlp_vocabulary.joblib', 'wb') as f:
            joblib.dump(vocab, f)
        with open('canon.joblib', 'wb') as f:
            joblib.dump(canon_table, f)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
