from typing import List
import numpy as np
from joinml.config import Config
from joinml.utils import get_sentence_transformer
from joinml.proxy.proxy import Proxy

from tqdm import tqdm
import numpy as np
import torch


class TransformerProxy(Proxy):
    def __init__(self, config: Config) -> None:
        self.model = get_sentence_transformer(config.proxy)
        self.device = config.device
        
    def get_tokenizer(self):
        return self.model.tokenize
    
    def get_proxy_score_for_tuples(self, tuples: List[List[str]]) -> np.ndarray:
        scores = []
        for tuple in tqdm(tuples):
            embeddings = self.model.encode(tuple)
            # cosine similarity
            scores.append(np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])))
        
        return scores
    
    def get_proxy_score_for_tables(self, table1: List[str], table2: List[str]) -> np.ndarray:
        embeddings1 = self.model.encode(table1, convert_to_tensor=True, device=self.device)
        embeddings2 = self.model.encode(table2, convert_to_tensor=True, device=self.device)
        print(embeddings1.shape)
        print(embeddings2.shape)
        # cosine similarity using torch in a pairwise manner
        scores = np.ones((len(table1), len(table2)))
        for i in tqdm(range(len(table1))):
            scores[i,:] = torch.cosine_similarity(embeddings1[i].unsqueeze(0), embeddings2).cpu().numpy()
        return scores


        