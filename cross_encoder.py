from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

class CrossEncoder:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-12-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def score(self, query, passages):
        # passages: list of strings
        pairs = [(query, passage) for passage in passages]
        inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        scores = logits.squeeze(-1).cpu().numpy()
        return scores
