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
    """Perform forward and backward translation with batching"""
    df[column] = df[column].astype(str)
    
    # Batch translate using DeepL
    try:
        translator = deepl.Translator(DEEPL_API_KEY)
        texts = df[column].tolist()
        translated = [result.text for result in translator.translate_text(texts, target_lang=target_lang_code_deepl)]
        df["Translated"] = translated
    except Exception as e:
        raise RuntimeError(f"DeepL translation failed: {str(e)}")

    # Batch back-translate using Google
    try:
        df["Back_Translated"] = _batch_google_translate(translated, target_lang_code_google)
    except Exception as e:
        raise RuntimeError(f"Google back-translation failed: {str(e)}")
    
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
