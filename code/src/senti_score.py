import ndjson
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from string import punctuation
from nltk.tokenize.casual import casual_tokenize
from src.nlp_preprocessing import clean_html_column
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from src.nlp_preprocessing import get_lemmas

def read_data(file_path: str) -> pd.DataFrame:
    """
    This function reads data from a file and returns it as a pandas DataFrame.
    It supports reading CSV, Excel, and JSON files.

    Parameters:
    - file_path (str): The path to the file to be read.

    Returns:
    - pd.DataFrame: The data read from the file as a pandas DataFrame.

    Raises:
    - FileNotFoundError: If the file does not exist.
    - ValueError: If the file format is not supported.
    """
    
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
        
    elif file_path.endswith('.json'):
        with open(file_path) as file:
            # Note: ndjson.load() can be used to read JSON files with newline delimited JSON (ndjson).
            df = ndjson.load(file)
            df = pd.DataFrame(df)
            
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    return df

def class_rebalancing(df: pd.DataFrame, rating_column: str,
                      sample_size: List[int] = [3_000, 1_000, 1_000, 1_000, 3_000], seed: int = 42) -> pd.DataFrame:
    """
    This function performs class rebalancing on a given DataFrame based on a specified rating column.
    It samples the data for each rating category according to the provided sample sizes and concatenates them into a new DataFrame.
    The resulting DataFrame is then exported to a CSV file at the specified export path.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the data to be rebalanced.
    - rating_column (str): The name of the column in the DataFrame that contains the rating values.
    - sample_size (List[int], optional): A list of integers representing the desired sample sizes for each rating category. Defaults to [3_000, 1_000, 1_000, 1_000, 3_000].
    - seed (int, optional): A random seed for reproducibility. Defaults to 42.

    Returns:
    - pd.DataFrame: The rebalanced DataFrame after sampling data for each rating category.

    Raises:
    - ValueError: If the rating column is not found in the input DataFrame.
    """
    rating_value = np.sort(df[rating_column].unique())

    one_star = df[df[rating_column] == rating_value[0]].sample(sample_size[0], random_state=seed)
    two_star = df[df[rating_column] == rating_value[1]].sample(sample_size[1], random_state=seed)
    three_star = df[df[rating_column] == rating_value[2]].sample(sample_size[2], random_state=seed)
    four_star = df[df[rating_column] == rating_value[3]].sample(sample_size[3], random_state=seed)
    five_star = df[df[rating_column] == rating_value[4]].sample(sample_size[4], random_state=seed)

    undersampled_df = pd.concat([one_star, two_star, three_star, four_star, five_star], axis=0)
    return undersampled_df
    
def data_preprocessing(df: pd.DataFrame, review_column: str, text_processing_method: str='stem') -> pd.DataFrame:
    """
    This function performs data preprocessing on a given DataFrame based on a specified review column.
    It cleans the HTML from the review column, converts the text to lowercase, removes punctuation, and tokenizes the text.
    Depending on the text_processing_method parameter, it either applies stemming or lemmatization to the tokens.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the data to be preprocessed.
    - review_column (str): The name of the column in the DataFrame that contains the review text.
    - text_processing_method (str, optional): A string representing the text processing method to be applied. Defaults to 'stem'.

    Returns:
    - pd.DataFrame: The preprocessed DataFrame after applying the specified text processing method.

    Raises:
    - ValueError: If the review_column is not found in the input DataFrame.
    - ValueError: If the text_processing_method is not 'stem' or 'lemma'.
    """
    df = clean_html_column(df, review_column, 'clean_review')

    df['clean_review'] = df['clean_review'].apply(lambda review:
        str(review).translate(str.maketranslate('', '', punctuation)).lower())

    df['tokens'] = df['clean_review'].apply(lambda review:
        casual_tokenize(str(review)))

    if text_processing_method == 'stem':
        stemmer = PorterStemmer()
        df['stemmed_tokens'] = df['tokens'].apply(lambda words:
            [stemmer.stem(word) for word in words])
    elif text_processing_method == 'lemma':
        lemmatizer = WordNetLemmatizer()
        df['lemma_tokens'] = df['tokens'].apply(lambda tokens:
            get_lemmas(tokens))
    else:
        raise ValueError(f"Unsupported text processing method: {text_processing_method}")

    return df
        
        