from joinml.config import Config
from joinml.dataset_loader import JoinDataset
from joinml.oracle import Oracle
from joinml.sampling.pilot_sample import get_evaluation_data
from joinml.proxy.proxy import Proxy
from joinml.utils import normalize

from itertools import product
import numpy as np
import logging
from tqdm import tqdm

class Evaluator:
    def __init__(self, config: Config, dataset: JoinDataset, oracle: Oracle):
        self.config = config
        self.eval_data_ids, self.eval_data_per_table = get_evaluation_data(config, dataset)
        self.positive_labels = set()
        self.is_self_join = config.is_self_join
        self.normalizing_factor = 1. / len(list(product(*self.eval_data_ids)))
        for row in tqdm(product(*self.eval_data_ids)):
            if oracle.query(row):
                self.positive_labels.add(row)
        num_rows = np.prod([len(ids) for ids in self.eval_data_ids])
        logging.info(f"positive rate of the evaluation data: {len(self.positive_labels) / num_rows}")        
        self.intrinsic_variance = np.var([1. for _ in range(len(self.positive_labels))] + [0. for _ in range(num_rows - len(self.positive_labels))])

    def evaluate(self, proxy: Proxy):
        if len(self.eval_data_per_table) == 2:
            scores = proxy.get_proxy_score_for_tables(self.eval_data_per_table[0], self.eval_data_per_table[1])
            scores = normalize(scores, is_self_join=self.is_self_join)
            assert np.abs(np.sum(scores) - 1) < 1e-5, f"{np.sum(scores)}"
            results = []
            for i in range(len(self.eval_data_ids[0])):
                for j in range(len(self.eval_data_ids[1])):
                    row = (self.eval_data_ids[0][i], self.eval_data_ids[1][j])
                    if row in self.positive_labels:
                        results.append(self.normalizing_factor/scores[i,j])
                    else:
                        results.append(0.)
            results = np.array(results)
            return (self.intrinsic_variance - np.var(results)) / self.intrinsic_variance
        else:
            raise NotImplementedError("Not implemented yet.")
