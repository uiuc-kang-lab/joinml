from typing import List
import numpy as np

class Proxy:
    def __init__(self) -> None:
        pass

    def get_proxy_score_for_tuples(self, tuples: List[List[str]]) -> np.ndarray:
        """Get the proxy score for a list of tuples."""
        pass

    def get_proxy_score_for_tables(self, table1: List[str], table2: List[str]) -> np.ndarray:
        """Get the proxy score for two tables."""
        pass