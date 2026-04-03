from sentence_transformers import SentenceTransformer
import numpy as np

# Load pre-trained SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')  # fast + good baseline

import os

documents = []
filenames = []

for filename in os.listdir("sec_filings"):
    path = os.path.join("sec_filings", filename)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
        documents.append(text)
        filenames.append(filename)

doc_embeddings = model.encode(documents, convert_to_numpy=True, show_progress_bar=True)

def search(query, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)[0]

    # Cosine similarity
    scores = np.dot(doc_embeddings, query_embedding) / (
        np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
    )

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = [(filenames[i], scores[i]) for i in top_indices]
    return results

results = search("apointment ceo's", top_k=3)

for file, score in results:
    print(file, score)

