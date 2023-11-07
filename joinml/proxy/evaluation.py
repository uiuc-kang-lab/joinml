from joinml.config import Config
from joinml.dataset_loader import JoinDataset
from joinml.oracle import Oracle
from joinml.sampling.pilot_sample import get_evaluation_data
from joinml.proxy.proxy import Proxy
from joinml.utils import normalize

from itertools import product
import numpy as np
import random

class Evaluator:
    def __init__(self, config: Config, dataset: JoinDataset):
        self.config = config
        if config.proxy_eval_data_sample_method == "random":
            ids = dataset.get_ids()
            if config.is_self_join:
                ids = [ids[0], ids[0]]
            all_tuples = list(product(*ids))
            sampled_tuples = random.sample(all_tuples, config.proxy_eval_data_sample_size)
            if config.is_self_join:
                self.sampled_ids = [[t[0], t[1]] for t in sampled_tuples if t[0] != t[1]]
                self.samples = [[dataset.id2join_col[0][t[0]], dataset.id2join_col[0][t[1]]] for t in self.sampled_ids]
            else:
                self.samples = []
                for t in sampled_tuples:
                    self.samples.append([dataset.id2join_col[i][t[i]] for i in range(len(t))])
        else:
            raise NotImplementedError(f"Sampler {config.proxy_eval_data_sample_method} is not implemented yet.")

    def evaluate(self, proxy: Proxy, oracle: Oracle):
        scores = proxy.get_proxy_score_for_tuples(self.samples)
        mse = .0
        for score, sample in zip(scores, self.sampled_ids):
            assert score >= 0 and score <= 1
            if oracle.query(sample):
                mse += (score - 1.) ** 2
            else:
                mse += score ** 2
        mse /= len(self.samples)
        return mse
