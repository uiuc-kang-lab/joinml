import cv2 as cv
import numpy as np
from typing import List
import numpy as np

from joinml.config import Config
from joinml.proxy.proxy import Proxy

"""
preprocess: 
1. load image from the path
2. convert the image to HSV
3. calculate image's histogram
4. normalize the histogram
"""
def _hist_process(path: str):
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    hist = cv.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256], accumulate=False)
    hist = cv.normalize(hist, hist)
    return hist

def _calculate_hist_comp(hist1: np.ndarray, hist2: np.ndarray):
    return (cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL) + 1) / 2

def _phash_process(path: str):
    img = cv.imread(path)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    h = cv.img_hash.pHash(img) # return array of 8 int
    h = int.from_bytes(h, byteorder='big', signed=False)
    # convert the base10 int to 32 bits 0/1 array
    h = np.array([int(i) for i in bin(h)[2:].zfill(64)])
    return h

"""
calculate the hamming distance between np ndarray
"""
def _calculate_hamming_distance(hash1: np.ndarray, hash2: np.ndarray):
    assert len(hash1) == len(hash2)
    return 1 - np.count_nonzero(hash1 != hash2) / len(hash1)


class OpencvProxy(Proxy):
    def __init__(self, config: Config) -> None:
        super().__init__()
        proxy_name = config.proxy

        if proxy_name == "Compare Histogram":
            self.preprocess = _hist_process
            self.sim_func = _calculate_hist_comp
        elif proxy_name == "pHash":
            self.preprocess = _phash_process
            self.sim_func = _calculate_hamming_distance

    """
    Calculates the similarity score between two tables.
    :param table1: List of paths to images
    :param table2: List of paths to images
    :return: 2D numpy array with similarity scores
    """
    def get_proxy_score_for_tables(self, table1: List[str], table2: List[str]) -> np.ndarray:
        hist1 = np.array([self.preprocess(path) for path in table1])
        hist2 = np.array([self.preprocess(path) for path in table2])
        scores = np.zeros((len(hist1), len(hist2)))
        for i in range(len(hist1)):
            for j in range(len(hist2)):
                scores[i][j] = self.sim_func(hist1[i], hist2[j])
        return scores

    """
    Calculates the similarity score between two images.
    :param tuples: List of lists of paths to images
    :return: 1D numpy array with similarity scores
    """
    def get_proxy_score_for_tuples(self, tuples: List[List[str]]) -> np.ndarray:
        hashes = np.array([(self.preprocess(t[0]), self.preprocess(t[1])) for t in tuples])
        scores = np.zeros(len(hashes))
        for i in range(len(tuples)):
            scores[i] = self.sim_func(hashes[i][0], hashes[i][1])
        return scores

if __name__ == "__main__":
    from itertools import product
    import time
    images = [f"../../data/city_vehicle/imgs/table0/{i}.jpg" for i in range(20)]
    table1 = images[:10]
    table2 = images[10:]
    tuples = list(product(images, images))
    config = Config()
    config.proxy = "Compare Histogram"
    proxy = OpencvProxy(config)
    start = time.time()
    for _ in range(10):
        proxy.get_proxy_score_for_tables(table1, table2)
        proxy.get_proxy_score_for_tuples(tuples)
    end = time.time()
    print(f"Time: {end-start}")

    config.proxy = "pHash"
    proxy = OpencvProxy(config)
    start = time.time()
    for _ in range(10):
        proxy.get_proxy_score_for_tables(table1, table2)
        proxy.get_proxy_score_for_tuples(tuples)
    end = time.time()
    print(f"Time: {end-start}")

# performance ranking
# run each method for 10 * (20*20 + 10*10) = 5000 pairs
# {'Compare Histogram': 0.44634079933166504, 'pHash': 0.5244479179382324} 