import sys
import pandas as pd
from sqlalchemy import create_engine

def get_categories(categories):
    '''
    Transform categories to one-hot encoded format.
    Args: categories (Pandas DataFrame)
    Returns: encoded categories
    '''
    categories = categories['categories'].str.split(';', expand=True)
    row = categories.iloc[[1]].values[0]
    categories.columns = [x.split("-")[0] for x in row]
    # convert to 1 or 0
    for col in categories:
        categories[col] = categories[col].map(lambda x: 1 if int(x.split("-")[1]) > 0 else 0)
    return categories

def load_data(messages_filepath, categories_filepath):
    '''
    Concate two loaded 2 datasets into one.
    Args: messages_filepath: file path to messages.csv 
          categories_filepath: file path to categories.csv
    Returns: Pandas DataFrame
    '''
    messages = pd.read_csv(messages_filepath)
    categories = get_categories(pd.read_csv(categories_filepath))
    return pd.concat([messages, categories], join="outer", axis=1)

def clean_data(df):
    # Drop duplicates of Pandas DataFrame
    return df.drop_duplicates()

def save_data(df, database_filename):
    # Save data to a disaster_response.db file
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_response', engine, if_exists='replace', index=False)

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