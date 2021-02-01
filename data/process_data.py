import sys
import os
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    ''' Function to load in messages and categories data,
        join them and return the result. This function also
        removes duplicates before merging datasets

        Args: 
            messages_filepath (string): path to the messages csv file.
            categories_filepath (string): path to the categories csv file

        Returns: 
            df (DataFrame): merged dataset containing messages and categories columns
            
    '''
    
    # read in messages data
    messages_df = pd.read_csv(messages_filepath)
    # remove duplicates
    messages_df.drop_duplicates(inplace=True)
    
    # read in categories data
    categories_df = pd.read_csv(categories_filepath)
    # remove duplicates
    categories_df.drop_duplicates(inplace=True)

    # merge the two datasets together based on id
    df = messages_df.merge(categories_df, on='id')
    
    return df


def clean_data(df):
    ''' Function that performs data cleaning on the messages + categories DataFrame.
        Function converts the categories column to individual columns by category,
        converts them to 0's and 1's, removes categories with only one value
        and finally removes duplicates in dataframes.

        Args: 
            df (DataFrame): merged dataset containing messages and categories columns

        Returns: 
            df (DataFrame): clean dataset containing messages and one binary (0 or 1) column per category
            
    '''
    
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
        
    # safeguard to make sure all categories have either 1 or 0
    categories = categories.clip(0, 1)
        
    # remove columns (cateories) where all rows have the same value (0, 1)
    single_categ_columns = [col for col in categories.columns if len(categories[col].unique()) == 1]
    categories = categories.drop(single_categ_columns, axis = 1)
        
    # drop categories column from original df
    df.drop(['categories'], axis=1, inplace=True)
    # concatenate the original dataframe with the new 'categories' dataframe
    df = pd.concat([df, categories], axis=1)
    
    # remove duplicates
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    ''' Function to save dataset into a database file locally.
        Dataset saved in a table named 'messages'.

        Args: 
            df (DataFrame): DataFrame to save to disk
            database_filename (string): path to database file to save the DataFrame in as csv

        Returns: 
            None
            
    '''
    
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