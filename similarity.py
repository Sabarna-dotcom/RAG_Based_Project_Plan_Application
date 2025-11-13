import json
import numpy as np
import requests
import joblib
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import os

np.random.seed(42)

# Data file paths
FAISS_PATH = "faiss.index"
CACHE_PATH = "vector_cache.pkl"


# Ollama embedding function
def embed_with_ollama(text_list, model="bge-m3"):
    """Generate embeddings using local Ollama."""
    r = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": model, "input": text_list}
    )
    r.raise_for_status()
    return r.json()["embeddings"]



class HybridSimilarity:
    def __init__(self, tfidf_max_features: int = 2000):
        self.tfidf = None
        self._corpus_meta = None
        self._corpus_texts = None
        self._corpus_emb = None
        self._faiss_index = None
        self.tfidf_max_features = tfidf_max_features
        self.cache_loaded = False


    @staticmethod
    def _build_text(op: Dict) -> str:
        # Combine fields into a single text blob
        tags = " ".join(op.get("tags", []))
        comps = " ".join(op.get("components", [])) if op.get("components") else ""
        title = op.get("title", "")
        desc = op.get("description", "")
        return f"{title}. {desc}. components: {comps}. tags: {tags}"


   
    # Load Joblib & FAISS
    def load_cache(self):
        if not (os.path.exists(CACHE_PATH) and os.path.exists(FAISS_PATH)):
            return False

        # Load Joblib
        cache = joblib.load(CACHE_PATH)

        self.tfidf = cache["tfidf"]
        self._tfidf_matrix = cache["tfidf_matrix"]
        self._corpus_emb = cache["embeddings"]
        self._corpus_meta = cache["metadata"]
        self._corpus_texts = cache["texts"]

        # Load FAISS
        self._faiss_index = faiss.read_index(FAISS_PATH)

        print("Loaded FAISS index & joblib cache.")
        self.cache_loaded = True
        return True


    
    # Save Joblib & FAISS
    def save_cache(self):
        cache = {
            "tfidf": self.tfidf,
            "tfidf_matrix": self._tfidf_matrix,
            "embeddings": self._corpus_emb,
            "metadata": self._corpus_meta,
            "texts": self._corpus_texts,
        }

        joblib.dump(cache, CACHE_PATH)
        faiss.write_index(self._faiss_index, FAISS_PATH)
        print("Saved FAISS index & joblib cache.")


    
    # Build vector db or load from database
    def fit(self, dataset_path: str = "dataset.json"):

        # If cache exists, load instead of recompute
        if self.load_cache():
            return

        # Load dataset
        with open(dataset_path, "r") as f:
            dataset = json.load(f)

        # Build texts for embedding & TF-IDF
        texts = [self._build_text(item) for item in dataset]

        # Save for cache
        self._corpus_meta = dataset
        self._corpus_texts = texts

        # TF-IDF
        self.tfidf = TfidfVectorizer(max_features=self.tfidf_max_features)
        self._tfidf_matrix = self.tfidf.fit_transform(texts)

        # Embeddings via Ollama
        emb = np.array(embed_with_ollama(texts), dtype="float32")
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        self._corpus_emb = emb

        # FAISS index
        dim = emb.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb)
        self._faiss_index = index

        # Save index + cache
        self.save_cache()


    
    # Query new opportunity
    def query(self, new_opportunity_path="new_opportunity.json", k=3, weight_emb=0.6, weight_tfidf=0.4):

        assert self._faiss_index is not None, "Call fit() first."

        # Load new opportunity
        with open(new_opportunity_path, "r") as f:
            new_op = json.load(f)

        new_text = self._build_text(new_op)

        # TF-IDF similarity
        q_tfidf = self.tfidf.transform([new_text])
        tfidf_scores = cosine_similarity(q_tfidf, self._tfidf_matrix)[0]

        # Embeddings via Ollama
        q_emb = np.array(embed_with_ollama([new_text]), dtype="float32")
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

        # FAISS Search 
        D, I = self._faiss_index.search(q_emb, len(self._corpus_emb))
        emb_scores_full = D[0]

        # Hybrid normalization
        matrix = np.vstack([emb_scores_full, tfidf_scores]).T
        scaled = MinMaxScaler().fit_transform(matrix)
        emb_scaled = scaled[:, 0]
        tfidf_scaled = scaled[:, 1]

        # Weighted final score
        final_scores = weight_emb * emb_scaled + weight_tfidf * tfidf_scaled

        # Top-k ranking
        top_idxs = np.argsort(final_scores)[-k:][::-1]

        # Build result
        return [
            {
                "id": self._corpus_meta[idx]["id"],
                "title": self._corpus_meta[idx]["title"],
                "relevance_score": float(round(final_scores[idx], 4))
            }
            for idx in top_idxs
        ]



# Manual test
if __name__ == "__main__":
    engine = HybridSimilarity()
    engine.fit("dataset.json")
    results = engine.query("new_opportunity.json")
    print(json.dumps(results, indent=2))
