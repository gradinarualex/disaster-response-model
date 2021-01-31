import re
import json
import plotly
import pandas as pd
import joblib

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine

from sklearn.base import BaseEstimator, TransformerMixin
    
app = Flask(__name__)

## Model Classes

def tokenize(text):
    clean_text = text.lower() # convert all chars to lower case
    clean_text = re.sub(r"[^a-zA-Z0-9]", " ", clean_text) # remove non alpha-numeric characters
    clean_text = re.sub(' +', ' ', clean_text) # remove duplicate spaces
    
    # tokenize text
    words = word_tokenize(clean_text)
    words = [w for w in words if w not in stopwords.words("english")]
    
    # reduce words to their stems
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(w) for w in words]
    
    # reduce words to root form
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(w) for w in stemmed]
    
    return clean_tokens

class QuestionMarkCount(BaseEstimator, TransformerMixin):
    '''
    '''

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # code here to transform data
        X_qcount = pd.Series(X).apply(lambda x: x.count('?'))

        return pd.DataFrame(X_qcount)

    
class ExclamationPointCount(BaseEstimator, TransformerMixin):
    '''
    '''

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # code here to transform data
        X_expointcount = pd.Series(X).apply(lambda x: x.count('!'))

        return pd.DataFrame(X_expointcount)

    
class CapitalCount(BaseEstimator, TransformerMixin):
    '''
    '''

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # code here to transform data
        X_capitalcount = pd.Series(X).apply(lambda text: sum(1 for c in text if c.isupper()))

        return pd.DataFrame(X_capitalcount)

    
class WordCount(BaseEstimator, TransformerMixin):
    '''
    '''

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # code here to transform data
        X_wordcount = pd.Series(X).apply(lambda x: len(x.split()))

        return pd.DataFrame(X_wordcount)

####


# load data
engine = create_engine('sqlite:///../data/disaster_response.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load(open("../models/disaster_response_model.pkl", 'rb'))


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()