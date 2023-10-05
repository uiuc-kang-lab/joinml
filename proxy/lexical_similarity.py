import py_stringmatching as sm
import csv
from typing import List, Literal
import multiprocessing
import sys
from tqdm import tqdm

class Cosine:
    def __init__(self) -> None:
        self.tokenizer = lambda l: l.split(" ")
        self.cos = sm.Cosine()

    def tokenize(self, table: List[str]):
        return [self.tokenizer(row) for row in table]
    
    def score_function(self, text1: List[str], text2: List[str]) -> float|Literal[0]:
        return self.cos.get_sim_score(text1, text2)
    
def run_job(args):
    id1, tokens1 = args
    results = []
    for id2, tokens2 in rtable:
        results.append([id1, id2, measure.score_function(tokens1, tokens2)])
    return results

def initialize_measure(method, _rtable):
    global measure
    global rtable
    rtable = _rtable
    if method == "cosine":
        measure = Cosine()
    else:
        raise NotImplementedError(f"{method} is not implemented")

def run_lexical_similarity_proxy(ltable: List[int|str], rtable: List[int|str], output_filename: str, method: str="cosine", num_worker: int="1"):
    ltable_text = [entry[1] for entry in ltable]
    ltable_id = [entry[0] for entry in ltable]
    rtable_text = [entry[1] for entry in rtable]
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
    scores = []
    # pbar = tqdm(total=len(ltable_tokens) * len(rtable_tokens))
    with multiprocessing.Pool(processes=num_worker, initializer=initialize_measure, initargs=(method, rtable_tokenized, )) as pool:
        for i, results in enumerate(pool.imap_unordered(run_job, ltable_tokenized)):
            scores += results
            sys.stderr.write('\rdone {0:%}'.format(i/len(ltable_tokens)))
    
    # output
    if output_filename.endswith(".csv"):
        with open(output_filename, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["id1", "id2", "score"])
            writer.writerows(scores)
    else:
        raise NotImplementedError(f"output method for file type {output_filename} is not implemented")