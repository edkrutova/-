# v4_1.py - BERT vectorizer
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class Vectorizer_BERT:
    def __init__(self, **params):
        model_name = params.get('model_name', 'distilbert-base-uncased')
        self.max_length = int(params.get('max_length', 128))
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        self.x_train_vector = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
    
    def _encode_texts(self, texts):
        vectors = []
        batch_size = 32
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            inputs = self.tokenizer(batch, padding=True, truncation=True, 
                                   max_length=self.max_length, return_tensors='pt')
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                vectors.append(batch_vectors)
        
        return np.vstack(vectors)
    
    def adapt(self, texts):
        self.x_train_vector = self._encode_texts(texts)
    
    def transform(self, texts):
        return self._encode_texts(texts)
    
    def train_vector(self):
        return self.x_train_vector
    
    def get_vocabulary(self):
        return ["BERT_embeddings"]
    
    def info(self):
        return f"vectorizer v4 [BERT]: max_length={self.max_length}, device={self.device}"
