import json
import plotly
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from plotly.graph_objs import Scatter

# from sklearn.externals import joblib
import joblib

from sqlalchemy import create_engine

from sklearn.model_selection import train_test_split

# import nltk
# nltk.download('punkt')
# nltk.download('wordnet')

import spacy
en_nlp = spacy.load('en')
stopwords = spacy.lang.en.stop_words.STOP_WORDS

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer('english')

app = Flask(__name__)

# https://realpython.com/natural-language-processing-spacy-python/ was very 
#	helpful as I struggled to build my first tokenizer.
def tokenize(text):
    # tokenize the text using spacy's model for English
    doc = en_nlp(text)
    # while we lemmatize the now tokenized text, let's not forget to drop
    #   tokens that are stop_words or punctuation
    lemmas = [token.lemma_ for token in doc
        if token not in stopwords and not token.is_punct]
    # Had better luck with this nltk stemmer
    stems = [stemmer.stem(lemma) for lemma in lemmas]
    return stems
	

def genF1PlotData(true, predicted):
	n = true.shape[1]
	result = np.empty(shape=n)
	for i in range(n):
		result[i] = f1_score(true[:,i], predicted[:,i], zero_division=0, average='micro')
	return result
	

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('MessageCategorization', engine)
# Originally I was going to compute the predicted values, but this takes several 
# minutes and we don't want our web server to be slow, so instead I'll read them from disk
with open('../models/predicted.joblib', 'rb') as f:
	y_predicted = joblib.load(f)


# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    y_names = list(df.columns)[4:]
    num_pos = df[y_names].sum()
    
    text = df['message'].values
    y = df[y_names].values
    text_train, text_test, y_train, y_test = train_test_split(text, y, \
    	test_size=0.33, random_state=42)

    print(y_test.shape)
    print(y_predicted.shape)
    
    f1_values = genF1PlotData(y_test, y_predicted)
    
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
                'title': 'Model Validity Vs. Inbalance',
                'yaxis': {
                    'title': "f1 Score"
                },
                'xaxis': {
                    'title': "Number of positives"
                }
            }
        },
                {
            'data': [
                Bar(
                    x=num_pos.index,
                    y=num_pos.values
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
#     print(*classification_labels)
#     print(*classification_results)

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()