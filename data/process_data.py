import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Ingests message and category data from files and returns a single
    dataframe holding the two parts properly merged.
    """
    # Load the two source .csv's into DataFrames
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Unpack categories column into separate columns
    categories = categories.categories.str.split(';', expand=True)

    # Take the first row from categories and extract the column headers
    row = categories.iloc[0]
    category_colnames = list(row.str.split('-', expand=True)[0])
    categories.columns = category_colnames

    # Extract the last character of each entry as the value
    for column in categories:
        # set value to be the last character of the string and cast it to int
        categories[column] = categories[column].str.slice(start=-1).astype(int)

    # Finally we have to merge the two datasets into one
    return pd.merge(
        left=messages, right=categories, left_index=True, right_index=True,
        copy=True)


def clean_data(df):
    """Removes any duplicates from the dataframe."""
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    """Constructs a database and writes the dataframe to it."""
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('MessageCategorization', engine, index=False,
              if_exists='replace')
    return


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath= sys.argv[1:]

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
