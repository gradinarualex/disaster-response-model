import sys
import os
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    # read in messages data
    messages_df = pd.read_csv('./data/messages.csv')
    # remove duplicates
    messages_df.drop_duplicates(inplace=True)
    
    # read in categories data
    categories_df = pd.read_csv('./data/categories.csv')
    # remove duplicates
    categories_df.drop_duplicates(inplace=True)

    # merge the two datasets together based on id
    df = messages_df.merge(categories_df, on='id')
    
    return df


def clean_data(df):
    
    # split categories into separate category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # get first row as sample to replace column names
    sample_row = categories.iloc[0,:]
    # clean column names
    category_colnames = sample_row.apply(lambda x: x.split('-')[0]).tolist()
    # set column names to the descriptive names created
    categories.columns = category_colnames
    
    # convert category values to just 0 and 1 numbers
    for column in categories.columns:
        # set the value to the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column to integer
        categories[column] = categories[column].astype(int)
        
    # drop categories column from original df
    df.drop(['categories'], axis=1, inplace=True)
    # concatenate the original dataframe with the new 'categories' dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates by grouping ID's and taking the maximum value (1 if any duplicate has 1 in column; 0 otherwise)
    df = df.groupby(['id', 'message', 'original', 'genre'], as_index=False).max()
    
    return df


def save_data(df, database_filename):
    # save data into sql database
    path = os.getcwd()
    db_url = 'sqlite:///' + database_filename

    # create database engine
    engine = create_engine(db_url)
    df.to_sql('messages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()