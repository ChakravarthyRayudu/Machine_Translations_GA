# utils.py
import json
import pandas as pd

def load_language_map(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def count_translatable_characters(df, column):
    """Count all characters in the column, including numbers and punctuation, skipping empty or NaN."""
    total_chars = 0
    for item in df[column]:
        if pd.isna(item) or str(item).strip() == '':
            continue
        total_chars += len(str(item))
    return total_chars
