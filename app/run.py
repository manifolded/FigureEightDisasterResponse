import json
import plotly
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Scatter
from plotly.graph_objs import Heatmap

import pickle as pkl
# from sklearn.externals import joblib
import joblib

import sqlalchemy as sqal

from sklearn.model_selection import train_test_split

import os

# import nltk
# nltk.download('punkt')
# nltk.download('wordnet')

app = Flask(__name__)


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


def gen_f1_plot_data(true, predicted):
    """Computes an f1 score for every target feature separately.

    INPUT:
        true - (pandas.DataFrame) actual y-values from test set
        pred - (pandas.DataFrame) predicted y-values for test set
    OUTPUT:
        f1_plot_data - (numpy.array) list of f1 scores
    """
    n = true.shape[1]
    result = np.empty(shape=n)
    for i in range(n):
        result[i] = f1_score(true[:,i], predicted[:,i], zero_division=0,
            average='micro', labels=[1])

    return result


# load data
engine = sqal.create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('MessageCategorization', engine)

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

# load model for hover notes on second plot
with open('../models/classifier.pkl', 'rb') as f:
    model = pkl.load(f)

# compute correlations for third plot
y_df = pd.DataFrame(data=y, columns=y_names)
corr_table = y_df.corr()


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
        },
        {
            'data': [
                Heatmap(
                    x=y_names,
                    y=y_names,
                    z=corr_table
                )
            ],
            'layout': {
                'title': 'Correlations Between Target Features in Labeled Samples',
                'yaxis': {
# https://stackoverflow.com/questions/55013995/plotly-heatmap-speed-and-aspect-ratio
                    'scaleanchor': 'x',
                    'autorange': 'reversed'
                },
                'xaxis': {
                },
# https://plotly.com/python/setting-graph-size/
                'autosize': True,
                'width': 700,
                'height': 700
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
