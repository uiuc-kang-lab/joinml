import csv
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

"""
assumption for the loaded table:
1. the first column is the id column
2. the last column is the join column
3. the id column starts from 0 and increments by 1
"""
class Table(object):
    def __init__(self, rows: List) -> None:
        self.id_col = np.array([int(row[0]) for row in rows])
    
    def get_ids(self):
        return self.id_col.tolist()

    def __len__(self):
        return len(self.id_col)
    

