import csv
from typing import List
import math
from sentence_transformers import SentenceTransformer
import numpy as np
import random
import numpy as np


def read_csv(path):
    """Read a CSV file."""
    with open(path) as f:
        reader = csv.reader(f)
        return list(reader)
    
def divide_sample_rate(rate: float, table_sizes: List[int]):
    table_rate = math.pow(rate, 1 / len(table_sizes))
    return [int(table_rate * size) for size in table_sizes]

def get_sentence_transformer(model_name: str="all-MiniLM-L6-v2") -> SentenceTransformer:
    return SentenceTransformer(model_name)

def normalize(x: np.ndarray, is_self_join: bool=False):
    # avoid select duplicate
    if is_self_join:
        assert len(x.shape) == 2 and x.shape[0] == x.shape[1]
        for i in range(x.shape[0]):
            x[i,i] = x.min()
    # normalize to [0, 1]
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    # avoid 0
    x[x == 0] = np.min(np.nonzero(x))
    # normalize to sum 1
    x /= np.sum(x)
    return x

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)