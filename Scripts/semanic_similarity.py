# semantic_similarity.py
from sentence_transformers import SentenceTransformer, util

# Initialize model
model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_semantic_similarity(original, translated):
    # Generate embeddings
    emb_original = model.encode(original, convert_to_tensor=True)
    emb_translated = model.encode(translated, convert_to_tensor=True)
    
    # Calculate cosine similarity
    similarity = util.pytorch_cos_sim(emb_original, emb_translated)
    return similarity.item()

# Example usage
if __name__ == "__main__":
    original = "The cat sat on the mat"
    translated = "A cat was sitting on a mat"
    
    score = calculate_semantic_similarity(original, translated)
    print(f"Semantic Similarity Score: {score:.3f}")
    

