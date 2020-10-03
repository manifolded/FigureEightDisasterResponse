import json
import plotly
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import en_core_web_sm

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Scatter

# from sklearn.externals import joblib
import joblib

import sqlalchemy as sqal

from sklearn.model_selection import train_test_split

import os

# import nltk
# nltk.download('punkt')
# nltk.download('wordnet')

import spacy
# en_nlp = spacy.load('en')
en_nlp = en_core_web_sm.load()
stopwords = spacy.lang.en.stop_words.STOP_WORDS

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')

app = Flask(__name__)


# https://realpython.com/natural-language-processing-spacy-python/ was very
#   helpful as I struggled to build this, my first, tokenizer.
def tokenize(text):
    """Takes a string containing a message and returns a list
    of tokens."""
    # tokenize the text using spacy's model for English
    doc = en_nlp(text)
    # while we lemmatize the now tokenized text, let's not forget to drop
    #   tokens that are stop_words or punctuation
    lemmas = [token.lemma_ for token in doc
        if token not in stopwords and not token.is_punct]
    # Had better luck with this nltk stemmer
    stems = [stemmer.stem(lemma) for lemma in lemmas]

    return stems


def gen_f1_plot_data(true, predicted):
    """Takes true and predicted y-values and computes an
    f1 score for every target category separately.
    """
    n = true.shape[1]
    result = np.empty(shape=n)
    for i in range(n):
        result[i] = f1_score(true[:,i], predicted[:,i], zero_division=0,
            average='micro', labels=[1])

    return result


# load data
engine = sqal.create_engine('sqlite:///../data/DisasterResponse.db')
print("DisasterResponse.db successfully opened.")
inspector = sqal.inspect(engine)
print("Schema Names: {}".format(inspector.get_schema_names()))  # -> ['main']
print("Table Names: {}".format(inspector.get_table_names(schema='main')))  # -> ['Message']
# inspector.get_columns('Message', schema='main')   # -> the goods

df = pd.read_sql_table('MessageCategorization', engine)

# correct outlying values in `related` column
### df['related'] = np.clip(df['related'], 0, 1)
df['related'].replace(to_replace=[2], value=0, inplace=True)

# drop 'child_alone' category which has no positives and thus breaks
#   LinearSVC
df.drop('child_alone', axis=1, inplace=True)

# extract data needed for visuals
y_names = list(df.columns)[4:]
num_pos = df[y_names].sum()

# Only compute f1 scores on test data, thus we must perform train/test
# split here.
text = df['message'].values
y = df[y_names].values
text_train, text_test, y_train, y_test = train_test_split(
    text, y, test_size=0.33, random_state=42)

# Originally I was going to compute the predicted values here, but
# this takes several minutes.  That would be much too slow for this
# flask app.  Instead, let's pre-compute those values in
# train_classifier.py and cache them to disk.  Here, we need just
# fetch them back.
with open('../models/predicted.joblib', 'rb') as f:
	y_predicted = joblib.load(f)
with open('../models/canon.joblib', 'rb') as f:
    canon_table = joblib.load(f)

# compute f1 scores for first plot
f1_values = gen_f1_plot_data(y_test, y_predicted)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # create visuals
    graphs = [
        {
            'data': [
                Scatter(
                    x=num_pos.values,
                    y=f1_values,
                    mode='markers'
                )
            ],

            'layout': {
                'title': 'Model Accuracy Vs. Inbalance',
                'yaxis': {
                    'title': "f1 Score"
                },
                'xaxis': {
                    'title': "Number of positives",
                    'type': "log"
                }
            }
        },
                {
            'data': [
                Bar(
                    x=num_pos.index,
                    y=num_pos.values,
                    text=canon_table
                )
            ],

            'layout': {
                'title': 'Number of Messages by Category in Dataset',
                'yaxis': {
                    'title': "Number of positives"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }

    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
#     app.run(host='0.0.0.0', port=3001, debug=True)
#    app.run(port=3001, debug=True)
    # Attempting this solution:
    # https://stackoverflow.com/questions/17260338/deploying-flask-with-heroku
    # Bind to PORT if defined, otherwise default to 3001.
    port = int(os.environ.get('PORT', 3001))
    app.run(host='0.0.0.0', port=port)



if __name__ == '__main__':
    main()
