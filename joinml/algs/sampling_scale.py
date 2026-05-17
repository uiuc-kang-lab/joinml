from joinml.proxy.get_proxy import get_proxy_score, get_proxy_rank
from joinml.scalable_dataset_loader import ScalableJoinDataset, load_dataset
from joinml.oracle import Oracle
from joinml.config import Config
from joinml.utils import set_up_logging, normalize
from joinml.estimates import Estimates
from joinml.utils import weighted_sample_pd, get_ci_bootstrap_ttest

import logging
import numpy as np
from typing import Tuple, List

def run(config: Config):
    set_up_logging(config.log_path, config.log_level)

    # log config
    logging.info(config)

    # dataset, oracle
    dataset = load_dataset(config)
    count_gt= dataset.get_gt()
    print(f"count gt: {count_gt}")
    
    for _ in range(config.internal_loop):

        if config.task == "uniform-scale":
            sample_results = dataset.wander_join(config.oracle_budget)
        else:
            sample_results = dataset.sample(0, config.oracle_budget, replace=True)

        sample_count = np.mean(sample_results) * dataset.get_stratum_size(0)
        
        est = Estimates(config.oracle_budget, dataset.get_gt(), sample_count, 0, 0)
        est.log()
        est.save(config.output_file, f"_{config.aggregator}")
