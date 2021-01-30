# import libraries
import sys
import os
import re

import pandas as pd
from sqlalchemy import create_engine
import sqlite3 as sql

import nltk
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


def load_data(database_filepath):
    '''
    '''
    
    # load data from database
    database_constring = 'sqlite:///' + database_filepath
    conn = create_engine(database_constring)
    df = pd.read_sql_table('messages', conn)

    labels = [col for col in df.columns if col not in ['id', 'message', 'original', 'genre', 'related']]

    # remove y labels with only one class (all 1's or all 0's) from output
    to_remove = []
    for label in labels:
        if df[label].sum() == 0:
            to_remove.append(label)

    labels = [label for label in labels if label not in to_remove]

    X = df['message'].values
    y = df[labels].values
    
    return X, y


def tokenize(text):
    '''
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
    '''
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
        ('clf', MultiOutputClassifier(GradientBoostingClassifier(max_depth=5, n_estimators=50, learning_rate=0.08)))
    ])
    
    # define parameters
    parameters = {
        'features__transformer_weights': [{'text_pipeline': 0.8, 'word_count': 0.05, 'qmark_count': 0.05, 'expoint_count': 0.05, 'capital_count': 0.05}]
    }

    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, y_test, category_names):
    '''
    '''
    
    # predict results over test set
    y_pred = model.predict(X_test)
    # create a multi-output evaluation report
    evaluation_report = classification_report(y_test, y_pred, target_names=labels)
    
    print(evaluation_report)

    return evaluation_report


def save_model(model, model_filepath):
    '''
    '''
    
    filename = 'disaster_response_model.pkl'
    pickle.dump(final_model, open(filename, 'wb'))


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