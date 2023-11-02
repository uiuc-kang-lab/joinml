import csv
from typing import List
import math
from sentence_transformers import SentenceTransformer
import numpy as np
import random
from numba import jit

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

@jit(nopython=True)
def calculate_score_for_tuples(embeddings: np.ndarray) -> np.ndarray:
    scores = np.zeros(len(embeddings))
    for i in range(len(embeddings)):
        scores[i] = np.dot(embeddings[i][0], embeddings[i][1]) / (np.linalg.norm(embeddings[i][0]) * np.linalg.norm(embeddings[i][1]))
    return scores

@jit(nopython=True)
def calculate_scre_for_tables(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    scores = np.ones((len(embeddings1), len(embeddings2)))
    for i in range(len(embeddings1)):
        for j in range(len(embeddings2)):
            scores[i][j] = np.dot(embeddings1[i], embeddings2[j]) / (np.linalg.norm(embeddings1[i]) * np.linalg.norm(embeddings2[j]))
    return scores