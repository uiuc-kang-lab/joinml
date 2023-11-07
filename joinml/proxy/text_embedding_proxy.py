from typing import List
import numpy as np
from joinml.config import Config
from joinml.utils import get_sentence_transformer
from joinml.proxy.proxy import Proxy
from joinml.utils import calculate_score_for_tuples, calculate_scre_for_tables

from tqdm import tqdm
import numpy as np
from itertools import product
import time


class TextEmbeddingProxy(Proxy):
    def __init__(self, config: Config) -> None:
        self.model = get_sentence_transformer(config.proxy)
        self.device = config.device
        
    def get_tokenizer(self):
        return self.model.tokenize

    def get_proxy_score_for_tuples(self, tuples: List[List[str]]) -> np.ndarray:
        embeddings = []
        left_tuples = [t[0] for t in tuples]
        right_tuples = [t[1] for t in tuples]
        left_embeddings = self.model.encode(left_tuples, device=self.device)
        right_embeddings = self.model.encode(right_tuples, device=self.device)
        for left_e, right_e in zip(left_embeddings, right_embeddings):
            embeddings.append([left_e, right_e])
        scores = calculate_score_for_tuples(np.array(embeddings))
        return scores
    
    def get_proxy_score_for_tables(self, table1: List[str], table2: List[str]) -> np.ndarray:
        embeddings1 = self.model.encode(table1, device=self.device)
        embeddings2 = self.model.encode(table2, device=self.device)
        scores = calculate_scre_for_tables(embeddings1, embeddings2)
        return scores

if __name__ == "__main__":
    sentences = [
        "How can Internet speed be increased by hacking through DNS?",
        "Which is the best digital marketing institution in banglore?",
        "Is it possible to store the energy of lightning?",
        "What is purpose of life?",
        "Why do Swiss despise Asians?",
        "What are the best associate product manager (APM) programs that someone in their early 20s can join to learn product management and have a rewarding career in the company?",
        "How can I speak with more clarity and confidence?",
        "How can we make the world a better place to live in for the future generations?",
        "How do you potty train a 4 months Pitbull?",
        "What will happen if I cancel TQWL tickets before the charting is done?",
        "How do people die?",
        "How can I keep my dog from getting fleas?",
        "How do I add a second device to a Google Play account?",
        "How did early Europeans get protein before the Columbian Exchange?",
        "Why can't we fall asleep on some days?",
        "How can I install OBB/data file on an Android?",
        "What are some good free web scrapers / scraping techniques?",
        "What are the best places to recommend foreigners to visit in Nepal?",
        "Why do some people believe that life ends at death?",
        "What is the future for iOS Developers?",
    ]
    table1 = sentences[:10]
    table2 = sentences[10:]
    tuples = list(product(sentences, sentences))

    config = Config()
    config.proxy = "all-MiniLM-L6-v2"
    config.device = "mps"
    proxy = TextEmbeddingProxy(config)

    start = time.time()
    print(proxy.get_proxy_score_for_tables(table1, table2))
    print(proxy.get_proxy_score_for_tuples(tuples))
    end = time.time()
    print(f"Time elapsed: {end - start}s (w/ compilation)")

    start = time.time()
    print(proxy.get_proxy_score_for_tables(table1, table2))
    print(proxy.get_proxy_score_for_tuples(tuples))
    end = time.time()
    print(f"Time elapsed: {end - start}s (pre-compilation)")