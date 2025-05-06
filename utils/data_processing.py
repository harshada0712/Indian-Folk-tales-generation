import pandas as pd
import re
import os
import nltk
from nltk.tokenize import sent_tokenize

def load_and_clean_data(file_path):
    """Your existing data loading and cleaning function"""
    # ... (copy the function from your notebook)
    return df, content_column

def prepare_datasets(df, content_column, test_size=0.1):
    """Your existing dataset preparation function"""
    # ... (copy the function from your notebook)
    return train_dataset, eval_dataset