# utils.py
import json
import pandas as pd
import os
import pandas as pd
from datetime import datetime

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

def create_backup_folder(original_filename):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(original_filename))[0]
    folder_name = f"{base_name}_{timestamp}"
    backup_path = os.path.join("backups", folder_name)
    os.makedirs(backup_path, exist_ok=True)
    return backup_path

def save_row_backup(row, backup_path, index):
    filename = f"row_{index}.csv"
    filepath = os.path.join(backup_path, filename)
    pd.DataFrame([row]).to_csv(filepath, index=False)
    return filepath


def list_backup_files(backup_path):
    return sorted(
        [f for f in os.listdir(backup_path) if f.startswith("row_")],
        key=lambda x: int(x.split('_')[1].split('.')[0])
    )


def list_backup_sessions():
    """List all backup session folders in the backups directory."""
    try:
        return [f for f in os.listdir("backups")
                if os.path.isdir(os.path.join("backups", f)) and not f.startswith(".")]
    except FileNotFoundError:
        return []
