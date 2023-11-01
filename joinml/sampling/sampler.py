import numpy as np
from typing import List
class RandomSampler:
    def __init__(self) -> None:
        pass

    def sample(self, data: np.ndarray, size: int, replace: bool=False) -> List[int]:
        """Sample `size` data points."""
        return np.random.choice(data, size, replace=replace)