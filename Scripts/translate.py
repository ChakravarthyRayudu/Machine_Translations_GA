# translation_functions.py (updated)
import deepl
import requests
import torch
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util
from config import DEEPL_API_KEY, GOOGLE_API_KEY

# Initialize model with device awareness
class SemanticModel:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            cls.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
            if device == 'cuda':
                cls.model = cls.model.half()
        return cls._instance


def perform_translation(df, column, target_lang_code_deepl, target_lang_code_google):
    """Perform forward and backward translation with batching. Handles empty values and logs errors."""

    df[column] = df[column].astype(str)

    # Skip empty/NaN and track original indices
    texts = df[column].tolist()
    translated_texts = []
    error_flags_deepl = []

    try:
        translator = deepl.Translator(DEEPL_API_KEY)
        for idx, text in enumerate(texts):
            if text.strip() == '' or text.lower() in ['nan', 'none']:
                translated_texts.append(text)
                error_flags_deepl.append(True)
                continue
            try:
                result = translator.translate_text(text, target_lang=target_lang_code_deepl)
                translated_texts.append(result.text)
                error_flags_deepl.append(False)
            except Exception as e:
                print(f"DeepL translation failed at index {idx}: {e}")
                translated_texts.append(text)
                error_flags_deepl.append(True)
    except Exception as e:
        print(f"Global DeepL failure: {e}")
        translated_texts = texts  # fallback to original
        error_flags_deepl = [True] * len(texts)

    df["Translated"] = translated_texts
    df["DeepL_Translation_Error"] = error_flags_deepl

    # Back-translation with Google
    back_translated = []
    error_flags_google = []

    try:
        for i in range(0, len(translated_texts), 50):
            batch = translated_texts[i:i+50]
            batch_results = []
            for j, text in enumerate(batch):
                if text.strip() == '' or text.lower() in ['nan', 'none']:
                    batch_results.append(text)
                    error_flags_google.append(True)
                    continue
                try:
                    params = {
                        'q': text,
                        'source': target_lang_code_google,
                        'target': 'en',
                        'key': GOOGLE_API_KEY
                    }
                    response = requests.post(
                        "https://translation.googleapis.com/language/translate/v2",
                        data=params
                    )
                    response.raise_for_status()
                    translation = response.json()['data']['translations'][0]['translatedText']
                    batch_results.append(translation)
                    error_flags_google.append(False)
                except Exception as e:
                    print(f"Google translation failed at index {i+j}: {e}")
                    batch_results.append(text)
                    error_flags_google.append(True)
            back_translated.extend(batch_results)
    except Exception as e:
        print(f"Global Google translation failure: {e}")
        back_translated = translated_texts  # fallback
        error_flags_google = [True] * len(translated_texts)

    df["Back_Translated"] = back_translated
    df["Google_Translation_Error"] = error_flags_google

    return df




def perform_evaluation(df, original_col="English", backtrans_col="Back_Translated"):
    """Add evaluation metrics with batch processing"""
    originals = df[original_col].astype(str).tolist()
    backtrans = df[backtrans_col].astype(str).tolist()
    
    # Semantic similarity (batched)
    semantic_model = SemanticModel().model
    emb_orig = semantic_model.encode(originals, convert_to_tensor=True, batch_size=32, show_progress_bar=False)
    emb_back = semantic_model.encode(backtrans, convert_to_tensor=True, batch_size=32, show_progress_bar=False)
    df["Semantic_Similarity"] = util.pytorch_cos_sim(emb_orig, emb_back).diag().cpu().numpy().round(4)
    
    # Text similarity
    df["Text_Similarity"] = [SequenceMatcher(None, o, b).ratio() for o, b in zip(originals, backtrans)]
    
    return df

def _batch_google_translate(texts, source_lang):
    translated = []
    for i in range(0, len(texts), 50):  # Batch in groups of 50
        batch = texts[i:i+50]
        params = {
            'q': batch,
            'source': source_lang,
            'target': 'en',
            'key': GOOGLE_API_KEY
        }
        response = requests.post(
            "https://translation.googleapis.com/language/translate/v2",
            data=params
        )
        response.raise_for_status()
        translated += [t['translatedText'] for t in response.json()['data']['translations']]
    return translated
