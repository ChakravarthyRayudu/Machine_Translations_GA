# text_diff.py
import difflib
from difflib import SequenceMatcher

def analyze_text_differences(original, translated):
    # Word-level diff
    differ = difflib.Differ()
    diff = list(differ.compare(original.split(), translated.split()))
    
    # Similarity ratio
    ratio = SequenceMatcher(None, original, translated).ratio()
    
    return diff, ratio

# Example usage
if __name__ == "__main__":
    original = "The cat sat on the mat"
    translated = "A cat was sitting on a mat"
    
    diff, ratio = analyze_text_differences(original, translated)
    
    print("Word-level Differences:")
    print('\n'.join(diff))
    print(f"\nSimilarity Ratio: {ratio:.3f}")
