import os
import pandas as pd
import deepl
import requests
from config import DEEPL_API_KEY, GOOGLE_API_KEY
from utils import create_backup_folder, save_row_backup

def perform_translation(df, column, target_lang_code_deepl, target_lang_code_google, original_filename):
    df[column] = df[column].astype(str)
    backup_path = create_backup_folder(original_filename)
    original_csv_path = os.path.join(backup_path, "original.csv")
    df.to_csv(original_csv_path, index=False)
    progress_file = os.path.join(backup_path, "progress.log")

    try:
        # DeepL Translation
        texts = df[column].tolist()
        translated_texts = []
        error_flags_deepl = []

        translator = deepl.Translator(DEEPL_API_KEY)
        for idx, text in enumerate(texts):
            try:
                if text.strip() in ('', 'nan', 'none'):
                    result_text = text
                    error_flag = True
                else:
                    result = translator.translate_text(text, target_lang=target_lang_code_deepl)
                    result_text = result.text
                    error_flag = False

                translated_texts.append(result_text)
                error_flags_deepl.append(error_flag)

                # Save partial backup after DeepL
                temp_df = df.iloc[[idx]].copy()
                temp_df["Translated"] = result_text
                temp_df["DeepL_Translation_Error"] = error_flag
                save_row_backup(temp_df.iloc[0], backup_path, idx)

            except Exception as e:
                translated_texts.append(text)
                error_flags_deepl.append(True)
                with open(progress_file, "a") as f:
                    f.write(f"DeepL Error at row {idx}: {str(e)}\n")

        df["Translated"] = translated_texts
        df["DeepL_Translation_Error"] = error_flags_deepl

        # Google Back-translation
        back_translated = []
        error_flags_google = []

        for i in range(0, len(translated_texts), 50):
            batch = translated_texts[i:i+50]
            batch_results = []
            for j, text in enumerate(batch):
                idx = i + j
                try:
                    if text.strip() in ('', 'nan', 'none'):
                        result_text = text
                        error_flag = True
                    else:
                        params = {
                            'q': text,
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
                        result_text = response.json()['data']['translations'][0]['translatedText']
                        error_flag = False

                    batch_results.append(result_text)
                    error_flags_google.append(error_flag)

                    # Update backup with back translation
                    row_path = os.path.join(backup_path, f"row_{idx}.csv")
                    row_df = pd.read_csv(row_path)
                    row_df["Back_Translated"] = result_text
                    row_df["Google_Translation_Error"] = error_flag
                    row_df.to_csv(row_path, index=False)

                except Exception as e:
                    batch_results.append(text)
                    error_flags_google.append(True)
                    with open(progress_file, "a") as f:
                        f.write(f"Google Error at row {idx}: {str(e)}\n")

            back_translated.extend(batch_results)

        df["Back_Translated"] = back_translated
        df["Google_Translation_Error"] = error_flags_google

        # Save final combined backup
        final_backup_path = os.path.join(backup_path, "FINAL.csv")
        df.to_csv(final_backup_path, index=False)

    except Exception as e:
        with open(progress_file, "a") as f:
            f.write(f"CRITICAL ERROR: {str(e)}\n")
        raise e

    return df
