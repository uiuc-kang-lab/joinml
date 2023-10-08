import csv
import numpy as np
from typing import List

"""
assumption for the loaded table:
1. the first column is the id column
2. the last column is the join column
3. the id column starts from 0 and increments by 1
"""
class Table(object):
    def __init__(self, rows: List) -> None:
        self.rows = rows        
        self.id_col = np.array([int(row[0]) for row in self.rows])
        self.join_col = np.array([row[-1] for row in self.rows])
    
    def get_join_col_by_ids(self, ids: List[int]) -> list:
        return self.join_col[ids]
    
    def get_ids(self):
        return self.id_col.tolist()

    def __len__(self):
        return len(self.rows)
    

