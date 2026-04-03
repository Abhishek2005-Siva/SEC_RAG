import os
import numpy as np
import re

docs = []

for filename in os.listdir("sec_filings"):
    file_path = os.path.join("sec_filings", filename)
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().lower()
        words = re.findall(r'\b[a-z]+\b', text)  # keep only clean words
        docs.append(words)
    
    
def tfidf(word,sentence):
    tf = sentence.count(word)
    idf = np.log10(len(docs) / sum([1 for doc in docs if word in doc])) if sum([1 for doc in docs if word in doc])!=0 else 0.0
    return round(tf*idf,4)

vocab = set(word for doc in docs for word in doc)

vectors = []
for x in docs:
    vec = []
    for word in vocab:
        vec.append(tfidf(word,x))
    vectors.append(vec)

