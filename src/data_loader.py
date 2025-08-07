import os

import pandas as pd


RAW_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "raw"
)


def load_raw_data(filename):
    """
    Load a CSV file from the data/raw directory.

    Args:
        filename (str): Name of the CSV file to load (e.g., 'mydata.csv')

    Returns:
        pd.DataFrame: Loaded data as a DataFrame
    Raises:
        FileNotFoundError: If the file does not exist
        pd.errors.EmptyDataError: If the file is empty
        pd.errors.ParserError: If the file cannot be parsed
    """
    file_path = os.path.join(RAW_DATA_DIR, filename)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        df = pd.read_csv(file_path)
        return df
    except pd.errors.EmptyDataError:
        print(f"Warning: The file {filename} is empty.")
        raise
    except pd.errors.ParserError as e:
        print(f"Error parsing {filename}: {e}")
        raise
