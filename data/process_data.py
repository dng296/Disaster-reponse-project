import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Function to load data from the messages and categories csv files, then combine them into one dataframe
    Args: messages_filepath, categories_filepath
    Returns: df - the combined dataset
    """
    # loading datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge the two datasets using the common id and save the combined dataframe for subsequent cleaning steps
    df = pd.merge(messages,categories, on=["id"])
    return df
    
    
def clean_data(df):
    """
    This function performs cleaning steps on the merged dataframe and return a cleaned df with:
       new separate columns for each category
       category values converted to just numbers 0 and 1 (suitable for ML)
       duplicates dropped
    """
    # split categories into separate columns
    categories = df['categories'].str.split(";", expand = True)
    # use the first row of categories dataframe to create column names, then rename the columns
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    # convert category values to just 0 and 1
    for column in categories:
        categories[column] = categories[column].str.slice(start=-1)
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original category column and replace with the new 'categories' df
    df.drop(['categories'], axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    # the 'related' column appears to contain values of 2, which will affect the ML step later, so we'll replace them with 1
    df['related'] = df['related'].replace(2,1)
    # remove duplicates
    df.drop_duplicates(inplace=True)
   
    return df
    

def save_data(df, database_filename):
    """
    Save the cleaned dataset into an sqlite database
    Args: cleaned dataframe (df) and database_filename to be saved
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Categorized_messages', engine, if_exists='replace', index=False)
    engine.dispose()


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