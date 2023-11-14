from joinml.proxy.get_proxy import get_proxy
from joinml.dataset_loader import JoinDataset
from joinml.oracle import Oracle
from joinml.config import Config
from joinml.utils import set_up_logging

import numpy as np
from scipy import stats
import logging

def run(config: Config):
    set_up_logging(config.log_path)

    # dataset, oracle
    dataset = JoinDataset(config)
    oracle = Oracle(config)

    # setup dataset
    dataset_ids = dataset.get_ids()
    join_cols = dataset.get_join_column_per_table(dataset_ids)
    if config.is_self_join:
        dataset_ids = [dataset_ids[0], dataset_ids[0]]
        join_cols = [join_cols[0], join_cols[0]]
    dataset_sizes = (len(dataset_ids[0]), len(dataset_ids[1]))
    gt = len(oracle.oracle_labels)

    for sample_size in config.lower_sample_size:
        errors = []
        for i in range(config.repeats):
            samples = np.random.choice(np.prod(dataset_sizes), size=sample_size, replace=False)
            samples_table_ids = np.array(np.unravel_index(samples, dataset_sizes)).T
            results = [] 
            for sample_table_id in samples_table_ids:
                if oracle.query(sample_table_id):
                    results.append(1.)
                else:
                    results.append(0.)
            
            results = np.array(results)
            ttest = stats.ttest_1samp(results, popmean=np.average(results))
            upper_bound = ttest.confidence_interval(confidence_level=config.confidence_level).high
            upper_bound_count = upper_bound * np.prod(dataset_sizes)
            error = (upper_bound_count - gt) / gt
            errors.append(error)
            logging.info(f"sample size {sample_size}/{i}, error= {error*100}%")
        
        errors = np.average(errors)
        std = np.std(errors)
        mean = np.average(errors)
        logging.info(f"sample size: average error= {mean*100}%, std= {std*100}%")


