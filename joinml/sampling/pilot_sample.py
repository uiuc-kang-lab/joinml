from joinml.config import Config
from joinml.oracle import Oracle
from joinml.dataset_loader import JoinDataset
from joinml.sampling.sampler import RandomSampler
from joinml.utils import divide_sample_rate

from typing import List, Tuple
import logging
import numpy as np


def get_evaluation_data(config: Config, dataset: JoinDataset) -> Tuple[List[List[int]], List[List[str]]]:
    """Get the data for proxy evaluation"""
    if config.proxy_eval_data_sample_method == "random":
        sampler = RandomSampler()
    else:
        raise NotImplementedError(f"Sampler {config.proxy_eval_data_sample_method} is not implemented yet.")
    
    table_sizes = dataset.get_sizes()
    # deal with self join
    if len(table_sizes) == 1:
        sample_size = int(np.sqrt(config.proxy_eval_data_sample_rate) * table_sizes[0])
        logging.info(f"training data sample sizes: {sample_size}")
        eval_data_ids = [sampler.sample(data=list(range(table_sizes[0])), size=sample_size, replace=True)]
        eval_data_per_table = dataset.get_join_column_per_table(ids=eval_data_ids)
        eval_data_ids *= 2
        eval_data_per_table *= 2
    else:
        # TODO: implement for non-self-join situation
        raise NotImplementedError("Not implemented yet.")
    
    return eval_data_ids, eval_data_per_table


