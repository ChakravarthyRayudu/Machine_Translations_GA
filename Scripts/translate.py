import os
import pandas as pd
import deepl
import requests
from config import DEEPL_API_KEY, GOOGLE_API_KEY
from utils import create_backup_folder

def perform_translation(df, column, target_lang_code_deepl, target_lang_code_google, original_filename):
    df = df.copy()  # Avoid modifying original input
    backup_path = create_backup_folder(original_filename)
    
    # Save original data
    original_csv_path = os.path.join(backup_path, "source.csv")
    df.to_csv(original_csv_path, index=False)
    
    # Add new columns
    extra_columns = ["Translated", "DeepL_Translation_Error", "Back_Translated", "Google_Translation_Error"]
    for col in extra_columns:
        df[col] = ""

    progress_file = os.path.join(backup_path, "progress.log")
    translated_csv_path = os.path.join(backup_path, "translated.csv")

    translator = deepl.Translator(DEEPL_API_KEY)

    # Create CSV with correct column order
    all_columns = [col for col in df.columns if col not in extra_columns] + extra_columns
    with open(translated_csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(f'"{col}"' for col in all_columns) + "\n")

    for idx, row in df.iterrows():
        original_text = str(row[column])
        
        # --- DeepL Translation ---
        translated = original_text
        deepl_error = True
        try:
            if original_text.strip().lower() not in ("", "nan", "none"):
                result = translator.translate_text(original_text, target_lang=target_lang_code_deepl)
                translated = result.text
                deepl_error = False
        except Exception as e:
            with open(progress_file, "a", encoding="utf-8") as f:
                f.write(f"DeepL Error at row {idx}: {str(e)}\n")

        # --- Google Back Translation ---
        back_translated = translated
        google_error = True
        try:
            if translated.strip().lower() not in ("", "nan", "none"):
                params = {
                    'q': translated,
                    'source': target_lang_code_google,
                    'target': 'en',
                    'key': GOOGLE_API_KEY
                }
                response = requests.post(
                    "https://translation.googleapis.com/language/translate/v2",
                    data=params,
                    timeout=10
                )
                response.raise_for_status()
                back_translated = response.json()['data']['translations'][0]['translatedText']
                google_error = False
        except Exception as e:
            with open(progress_file, "a", encoding="utf-8") as f:
                f.write(f"Google Error at row {idx}: {str(e)}\n")

        # Update DataFrame
        df.at[idx, "Translated"] = translated
        df.at[idx, "DeepL_Translation_Error"] = deepl_error
        df.at[idx, "Back_Translated"] = back_translated
        df.at[idx, "Google_Translation_Error"] = google_error

        # Prepare row in correct order for saving
        updated_row = df.loc[idx, all_columns].tolist()
        with open(translated_csv_path, "a", encoding="utf-8") as f:
            f.write(",".join(f'"{str(item).replace("\"", "\"\"")}"' for item in updated_row) + "\n")

    # Return final DataFrame with reordered columns
    df = df[all_columns]
    return df
