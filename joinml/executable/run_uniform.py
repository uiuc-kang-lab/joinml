from joinml.dataset_loader import JoinDataset
from joinml.oracle import Oracle
from joinml.config import Config
from joinml.utils import set_up_logging, get_ci_gaussian, get_ci_ttest

import numpy as np
import random
from scipy import stats
import logging

def run(config: Config):
    set_up_logging(config.log_path)

    # dataset, oracle
    dataset = JoinDataset(config)
    oracle = Oracle(config)

    # setup dataset
    dataset_sizes = dataset.get_sizes()
    if config.is_self_join:
        dataset_sizes = (dataset_sizes[0], dataset_sizes[0])
    gt = len(oracle.oracle_labels)

    logging.info(f"ground truth: {gt}")

    if isinstance(config.sample_size, int):
        config.sample_size = [config.sample_size]

    full_size = np.prod(dataset_sizes)
    for sample_size in config.sample_size:
        count_results = []
        true_errors = []
        gaussian_upper_errors = []
        ttest_upper_errors = []
        for i in range(config.repeats):
            samples = np.random.choice(full_size, size=sample_size, replace=True)
            # samples = random.choices(list(range(full_size)), k=sample_size)
            samples_table_ids = np.array(np.unravel_index(samples, dataset_sizes)).T
            results = [] 
            for sample_table_id in samples_table_ids:
                if oracle.query(sample_table_id):
                    results.append(1.)
                else:
                    results.append(0.)
            
            results = np.array(results)
            count_result = results.mean() * full_size
            true_error = (count_result - gt) / gt
            # get estimated confidence interval
            if count_result != 0:
                _, gaussian_upper = get_ci_gaussian(results, config.confidence_level)
                gaussian_upper *= full_size
                gaussian_upper_error = (gaussian_upper - gt) / gt
                _, ttest_upper = get_ci_ttest(results, config.confidence_level)
                ttest_upper_error = (ttest_upper*full_size - gt) / gt
                ttest_upper *= full_size
                gaussian_upper_errors.append(gaussian_upper_error)
                ttest_upper_errors.append(ttest_upper_error)
                logging.info(f"sample size {sample_size} trial {i} count result {count_result} true error {true_error} gaussian upper {gaussian_upper} gaussian upper error {gaussian_upper_error} ttest upper {ttest_upper} ttest upper error {ttest_upper_error}")
            else:
                logging.info(f"sample size {sample_size} trial {i} count result {count_result} true error {true_error}")
            count_results.append(count_result)
            true_errors.append(true_error)

        average_count_result = np.mean(count_results)
        average_true_error = np.mean(true_errors)
        std_true_error = np.std(true_errors)
        gaussian_upper_error = np.mean(gaussian_upper_errors)
        ttest_upper_error = np.mean(ttest_upper_errors)
        logging.info(f"results: sample size {sample_size} average count result {average_count_result} average true error {average_true_error} std true error {std_true_error} gaussian upper error {gaussian_upper_error} ttest upper error {ttest_upper_error}")


