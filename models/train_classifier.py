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

en_nlp = spacy.load('en')
stopwords = spacy.lang.en.stop_words.STOP_WORDS
stemmer = SnowballStemmer('english')

def genTable(true, pred, zero_division='warn'):
    table = np.empty(shape=(true.shape[1], 3))
    for i in range(true.shape[1]):
        table[i] = \
            [precision_score(true[:,i], pred[:,i], zero_division=zero_division),
            recall_score(true[:,i], pred[:,i], zero_division=zero_division),
            f1_score(true[:,i], pred[:,i], zero_division=zero_division)]
    return table


def printScores(labels, true, pred, zero_division='warn'):
    results = genTable(true, pred, zero_division=zero_division)
    print("{0:>20}".format(''), 'prec', 'recall', 'f1', sep='\t')
    for i in range(results.shape[0]):
        print("{0:>20}".format(labels[i]), '%.2f' % results[i, 0],
            '%.2f' % results[i,1], '%.2f' % results[i,2], sep = '\t')


def load_data(database_filepath):
    # open the database file created by previous script
    engine = sqal.create_engine('sqlite:///' + database_filepath)
    # and grab the table therein
    df = pd.read_sql_table('MessageCategorization', engine)

    in_columns = 'message'
    out_columns = list(df.columns)[5:]

    # remove outliers from 'related' column
    df['related'] = np.clip(df['related'], 0, 1)

    text = df[in_columns].values
    y = df[out_columns].values

    # # save some for data for testing the trained model
    # text_train, text_test, y_train, y_test = \
    #     train_test_split(text, y, test_size=0.33, random_state=42)

    return text, y, out_columns


# https://realpython.com/natural-language-processing-spacy-python/ was helpful
def tokenize(text):
    # tokenize the text using spacy's model for English
    doc = en_nlp(text)
    # while we lemmatize the now tokenized text, let's not forget to drop
    #   tokens that are stop_words or punctuation
    lemmas = [token.lemma_ for token in doc
        if token not in stopwords and not token.is_punct]
    # Had better luck with this nltk stemmer
    return [stemmer.stem(lemma) for lemma in lemmas]


def build_model():
    model = make_pipeline(
        TfidfVectorizer(tokenizer=tokenize, min_df=5),
        MultiOutputClassifier(
            estimator=AdaBoostClassifier(
                base_estimator=DecisionTreeClassifier(max_depth=2),
                n_estimators=10, learning_rate=1)))
    return model


def evaluate_model(model, text_test, y_test, category_names):
    y_pred = model.predict(text_test)
    print('Test data cv score = {0:.2f}'.format(model.score(text_test, y_test)))
    printScores(category_names, y_test, y_pred, zero_division=0)


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pkl.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
