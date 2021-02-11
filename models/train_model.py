# import libraries
import sys
import os
import re
import joblib

import pandas as pd
from sqlalchemy import create_engine
import sqlite3 as sql

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk import pos_tag, ne_chunk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

from sklearn.metrics import classification_report


class QuestionMarkCount(BaseEstimator, TransformerMixin):
    ''' Class that creates a new feature for natural language
        processing prediciton model. This class counts the 
        number of question marks in a text.
        
        Functions: 
            fit: returns self
            transform: count the number of question marks and return as DataFrame
            
    '''

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # code here to transform data
        X_qcount = pd.Series(X).apply(lambda x: x.count('?'))

        return pd.DataFrame(X_qcount)

    
class ExclamationPointCount(BaseEstimator, TransformerMixin):
    ''' Class that creates a new feature for natural language
        processing prediciton model. This class counts the 
        number of excalamtion points in a text.
        
        Functions: 
            fit: returns self
            transform: count the number of excalamtion point and return as DataFrame
            
    '''

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # code here to transform data
        X_expointcount = pd.Series(X).apply(lambda x: x.count('!'))

        return pd.DataFrame(X_expointcount)

    
class CapitalCount(BaseEstimator, TransformerMixin):
    ''' Class that creates a new feature for natural language
        processing prediciton model. This class counts the 
        number of upper case letters in a text.
        
        Functions: 
            fit: returns self
            transform: count the number of upper case letters and return as DataFrame
            
    '''

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # code here to transform data
        X_capitalcount = pd.Series(X).apply(lambda text: sum(1 for c in text if c.isupper()))

        return pd.DataFrame(X_capitalcount)

    
class WordCount(BaseEstimator, TransformerMixin):
    ''' Class that creates a new feature for natural language
        processing prediciton model. This class counts the 
        number of words in a text.
        
        Functions: 
            fit: returns self
            transform: count the number of words and return as DataFrame
            
    '''

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # code here to transform data
        X_wordcount = pd.Series(X).apply(lambda x: len(x.split()))

        return pd.DataFrame(X_wordcount)


def load_data(database_filepath):
    ''' Function used to load data from a database using sql_engine to access
        an sqlite database file and return the values as features, targets
        and category labels.

        Args: 
            database_filepath (string): path to a database to read in 
            the data from the 'messages' table

        Returns: 
            X (array): array of features
            y (array): array of target columns
            labels (list): list of label strings for each y column

    '''
    
    # load data from database
    database_constring = 'sqlite:///' + database_filepath
    conn = create_engine(database_constring)
    df = pd.read_sql_table('messages', conn)

    X = df['message'].values
    y = df.iloc[:, 4:].values
    
    return X, y, df.iloc[:, 4:].columns


def tokenize(text):
    ''' Function to tokenize text.
        This function converts all text to lower case, removes
        non-alpha-numeric characters, removes duplicate spaces,
        tokenizes, stems and lemmatizes words.

        Args: 
            text (string): path to a database to read in

        Returns: 
            clean_tokens (list): list of tokenized words
            
    '''
    
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


def build_model():
    ''' Function to build a text classification model.
        Build pipeline, defines parameters, creates and
        returns Cross Validation model.

        Args: 
            None

        Returns: 
            cv (model): cross-validation classifier model
            
    '''
    
    # build pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('textpipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize, ngram_range=(1,2))),
                ('tfidf', TfidfTransformer(smooth_idf=False)),
            ])),
            ('qmark_count', QuestionMarkCount()),
            ('expoint_count', ExclamationPointCount()),
            ('capital_count', CapitalCount()),
            ('word_count', WordCount())
        ])),
        ('clf', MultiOutputClassifier(GradientBoostingClassifier(max_depth=8, n_estimators=100, learning_rate=0.07)))
    ])
    
    # define parameters
    parameters = {
        'features__transformer_weights': [{'text_pipeline': 0.9, 'word_count': 0.025, 'qmark_count': 0.025, 'expoint_count': 0.025, 'capital_count': 0.025}]
    }

    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    ''' Function to evaluate the performance of our model.
        Uses classification_report to get recall, precision
        and f1-score by category and some overall averages.
        Build pipeline, defines parameters, creates and
        returns Cross Validation model.

        Args: 
            model: the classification model to be evaluated
            X_test: test set of features to predict on
            y_test: correct category values for test set
            category_names: list of strings representing category names

        Returns: 
            evaluation_report: report showing model scores
    '''
    
    # predict results over test set
    y_pred = model.predict(X_test)
    # create a multi-output evaluation report
    evaluation_report = classification_report(y_test, y_pred, target_names=category_names)
    
    print(evaluation_report)

    return evaluation_report


def save_model(model, model_filepath):
    ''' Function to save the fitted model to disk.
    
        Args: 
            model: model to be saved on disk
            model_filepath: file path to where you want to save the model

        Returns: 
            None
            
    '''
    
    filename = 'disaster_response_model.sav'
    joblib.dump(model, filename)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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