import py_stringmatching as sm
import csv
from typing import List, Literal
import multiprocessing
import sys
from tqdm import tqdm
import logging

class Cosine:
    def __init__(self) -> None:
        self.tokenizer = lambda l: l.split(" ")
        self.cos = sm.Cosine()

    def tokenize(self, table: List[str]):
        return [self.tokenizer(row) for row in table]
    
    def score_function(self, text1: List[str], text2: List[str]) -> float|Literal[0]:
        return self.cos.get_sim_score(text1, text2)
    
class Process(object):
    def __init__(self, method: str, rtable: List, output_filename: str) -> None:
        # load methods
        self.method = method
        self.rtable = rtable
        # self.lock = lock
        self.output_filename = output_filename
    
    def __call__(self, args):
        if self.method == "cosine":
            measure = Cosine()
        else:
            raise NotImplementedError(f"{self.method} is not implemented")
        
        id1, tokens1 = args
        results = []
        for id2, tokens2 in self.rtable:
            results.append([id1, id2, f"{measure.score_function(tokens1, tokens2):.2}"])
        
        # self.lock.acquire()
        with open(self.output_filename, "a+") as f:
            writer = csv.writer(f)
            writer.writerow(["id1", "id2", "score"])
            writer.writerows(results)
        logging.info(f"finish id {id1}")
        # self.lock.release()

def run_lexical_similarity_proxy(ltable: List[int|str], rtable: List[int|str], output_filename: str, method: str="cosine", num_worker: int="1"):
    ltable_text = [entry[-1] for entry in ltable]
    ltable_id = [entry[0] for entry in ltable]
    rtable_text = [entry[-1] for entry in rtable]
    rtable_id = [entry[0] for entry in rtable]

    # load methods
    if method == "cosine":
        measure = Cosine()
    else:
        raise NotImplementedError(f"{method} is not implemented")

    # calculate scores
    ltable_tokens = measure.tokenize(ltable_text)
    rtable_tokens = measure.tokenize(rtable_text)
    ltable_tokenized = [[lid, rtokens] for lid, rtokens in zip(ltable_id, ltable_tokens)]
    rtable_tokenized = [[rid, rtokens] for rid, rtokens in zip(rtable_id, rtable_tokens)]
    # pbar = tqdm(total=len(ltable_tokens) * len(rtable_tokens))
    lock = multiprocessing.Lock()
    with multiprocessing.Pool(processes=num_worker) as pool:
        pool.map(Process(method, rtable_tokenized, output_filename), ltable_tokenized)
