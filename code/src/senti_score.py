import ndjson
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

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

def class_rebalancing(df: pd.DataFrame, rating_column: str, export_path: str,
                      sample_size: List[int] = [3_000, 1_000, 1_000, 1_000, 3_000], seed: int = 42) -> None:
    """
    This function performs class rebalancing on a given DataFrame based on a specified rating column.
    It samples the data for each rating category according to the provided sample sizes and concatenates them into a new DataFrame.
    The resulting DataFrame is then exported to a CSV file at the specified export path.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the data to be rebalanced.
    - rating_column (str): The name of the column in the DataFrame that contains the rating values.
    - export_path (str): The path to the CSV file where the rebalanced data will be exported.
    - sample_size (List[int], optional): A list of integers representing the desired sample sizes for each rating category.
                                        Defaults to [3_000, 1_000, 1_000, 1_000, 3_000].
    - seed (int, optional): A random seed for reproducibility. Defaults to 42.

    Returns:
    - None: This function does not return any value, but it modifies the input DataFrame and exports the rebalanced data to a CSV file.

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
    undersampled_df.to_csv(export_path, index=False)