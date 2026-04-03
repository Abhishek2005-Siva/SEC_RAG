from rank_bm25 import BM25Okapi
import os

folder_path = "sec_filings"

documents = []
filenames = []

for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            documents.append(f.read().lower())
            filenames.append(filename)

# BM25
bm25 = BM25Okapi([doc.split() for doc in documents])

query = "appointment chief 5.02"
query_score = bm25.get_scores(query.split())

# Print filename with score
for fname, score in zip(filenames, query_score):
    print(f"{fname} --> {score:.4f}")
    