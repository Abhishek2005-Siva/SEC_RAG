import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "Dogs are allowed in the office on Fridays",
    "Pets can come to work on Furry days",
    "Remote work policies allow 3 days from home"
]

embeddings = model.encode(sentences)

sim_1_2 = np.dot(embeddings[0],embeddings[1])
sim_1_3 = np.dot(embeddings[0],embeddings[2])
sim_2_3 = np.dot(embeddings[1],embeddings[2])

print(sim_1_2)
print(sim_1_3)
print(sim_2_3)
