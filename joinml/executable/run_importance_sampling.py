from joinml.proxy.get_proxy import get_proxy
from joinml.dataset_loader import JoinDataset
from joinml.oracle import Oracle
from joinml.config import Config
from joinml.utils import set_up_logging, get_ci_gaussian, get_ci_ttest, preprocess, normalize

import numpy as np
import logging
import os

def run(config: Config):
    set_up_logging(config.log_path)

    # dataset, oracle
    dataset = JoinDataset(config)
    oracle = Oracle(config)

    # setup dataset
    dataset_sizes = dataset.get_sizes()
    if config.is_self_join:
        dataset_sizes = [dataset_sizes[0], dataset_sizes[0]]
    gt = len(oracle.oracle_labels)

    logging.info(f"ground truth: {gt}")

    # check cache
    proxy_store_path = f"{config.cache_path}/{config.dataset_name}_{config.proxy.split('/')[-1]}_scores.npy"
    if config.proxy_score_cache and os.path.exists(proxy_store_path):
        logging.info("Loading proxy scores from %s", proxy_store_path)
        proxy_scores = np.load(proxy_store_path)
        assert np.prod(proxy_scores.shape) == np.prod(dataset_sizes), "Proxy scores shape does not match dataset sizes."
    else:
        logging.info("Calculating proxy scores.")
        proxy = get_proxy(config)
        join_columns = dataset.get_join_column()
        if config.is_self_join:
            join_columns = [join_columns[0], join_columns[0]]
        proxy_scores = proxy.get_proxy_score_for_tables(join_columns[0], join_columns[1])
        logging.info("Postprocessing proxy scores.")
        proxy_scores = preprocess(proxy_scores, is_self_join=config.is_self_join)
        proxy_scores = proxy_scores.flatten()

        if config.proxy_score_cache:
            logging.info("Saving proxy scores to %s", proxy_store_path)
            np.save(proxy_store_path, proxy_scores)

    if isinstance(config.sample_size, int):
        config.sample_size = [config.sample_size]

    proxy_scores = normalize(proxy_scores, style=config.proxy_normalizing_style)

    for sample_size in config.sample_size:
        count_results = []
        true_errors = []
        gaussian_upper_errors = []
        ttest_upper_errors = []
        for i in range(config.repeats):
            samples = np.random.choice(np.prod(dataset_sizes), size=sample_size, replace=True, p=proxy_scores)
            samples_table_ids = np.array(np.unravel_index(samples, dataset_sizes)).T
            results = [] 
            for sample_table_id, sample in zip(samples_table_ids, samples):
                if oracle.query(sample_table_id):
                    results.append(1. / len(np.prod(dataset_sizes)) / proxy_scores[sample])
                else:
                    results.append(0.)
            
            results = np.array(results)
            count_result = results.mean() * np.prod(dataset_sizes)
            true_error = (count_result - gt) / gt
            # get estimated confidence interval
            if count_result != 0:
                gaussian_upper, _ = get_ci_gaussian(results, config.confidence)
                gaussian_upper_error = (gaussian_upper - gt) / gt
                ttest_upper, _ = get_ci_ttest(results, config.confidence)
                ttest_upper_error = (ttest_upper - gt) / gt
                gaussian_upper_errors.append(gaussian_upper_error)
                ttest_upper_errors.append(ttest_upper_error)
                logging.info(f"sample size {sample_size} trial {i} count result {count_result} true error {true_error} gaussian upper {gaussian_upper} gaussian upper error {gaussian_upper_error} ttest upper {ttest_upper} ttest upper error {ttest_upper_error}")

            logging.info(f"sample size {sample_size} trial {i} count result {count_result} true error {true_error}")
            count_results.append(count_result)
            true_errors.append(true_error)

        average_count_result = np.mean(count_results)
        average_true_error = np.mean(true_errors)
        std_true_error = np.std(true_errors)
        gaussian_upper_error = np.mean(gaussian_upper_errors)
        ttest_upper_error = np.mean(ttest_upper_errors)
        logging.info(f"results: sample size {sample_size} average count result {average_count_result} average true error {average_true_error} std true error {std_true_error} gaussian upper error {gaussian_upper_error} ttest upper error {ttest_upper_error}")


