from joinml.proxy.get_proxy import get_proxy
from joinml.dataset_loader import JoinDataset
from joinml.oracle import Oracle
from joinml.config import Config
from joinml.utils import set_up_logging, get_ci_gaussian, get_ci_ttest, preprocess, normalize, defensive_mix

import numpy as np
import logging
import os

def run(config: Config):
    set_up_logging(config.log_path)

    # log config
    logging.info(config)

    # dataset, oracle
    dataset = JoinDataset(config)
    oracle = Oracle(config)

    # setup dataset
    dataset_sizes = dataset.get_sizes()
    if config.is_self_join:
        dataset_sizes = (dataset_sizes[0], dataset_sizes[0])
    gt = len(oracle.oracle_labels)

    logging.info(f"ground truth: {gt}")

    # check cache for proxy scores
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

    # check cache for proxy rank
    proxy_rank_store_path = f"{config.cache_path}/{config.dataset_name}_{config.proxy.split('/')[-1]}_rank.npy"
    if config.blocking_cache and os.path.exists(proxy_rank_store_path):
        logging.info("Loading proxy rank from %s", proxy_rank_store_path)
        proxy_rank = np.load(proxy_rank_store_path)
        assert np.prod(proxy_rank.shape) == np.prod(dataset_sizes), "Proxy rank shape does not match dataset sizes."
    else:
        logging.info("Calculating proxy rank.")
        proxy_rank = np.argsort(proxy_scores)
        if config.blocking_cache:
            logging.info("Saving proxy rank to %s", proxy_rank_store_path)
            np.save(proxy_rank_store_path, proxy_rank)

    if isinstance(config.sample_size, int):
        config.sample_size = [config.sample_size]

    if isinstance(config.blocking_size, int):
        config.blocking_size = [config.blocking_size]

    for blocking_size in config.blocking_size:
        logging.info(f"running with blocking size: {blocking_size}")
        unblocked_samples = proxy_rank[-blocking_size:]
        blocked_samples = proxy_rank[:-blocking_size]
        blocked_proxy_scores = proxy_scores[blocked_samples]
        blocked_proxy_scores = normalize(blocked_proxy_scores, style=config.proxy_normalizing_style)
        blocked_proxy_scores = defensive_mix(blocked_proxy_scores, ratio=config.defensive_rate)

        # run oracle on the unblocked samples
        unblocked_samples_table_ids = np.array(np.unravel_index(unblocked_samples, dataset_sizes)).T
        unblocked_positives = 0
        for unblocked_sample_table_id in unblocked_samples_table_ids:
            if oracle.query(unblocked_sample_table_id):
                unblocked_positives += 1
        logging.info(f"unblocked positives: {unblocked_positives}")
        logging.info(f"blocked positives: {gt - unblocked_positives}")

        for sample_size in config.sample_size:
            count_results = []
            true_errors = []
            gaussian_upper_errors = []
            ttest_upper_errors = []
            for i in range(config.repeats):
                samples = np.random.choice(len(blocked_samples), size=sample_size, replace=True, p=blocked_proxy_scores)
                sample_ids = blocked_samples[samples]
                samples_table_ids = np.array(np.unravel_index(sample_ids, dataset_sizes)).T
                results = [] 
                for sample_table_id, sample in zip(samples_table_ids, samples):
                    if oracle.query(sample_table_id):
                        results.append(1. / len(blocked_samples) / blocked_proxy_scores[sample])
                    else:
                        results.append(0.)
                
                results = np.array(results)
                count_result = results.mean() * len(blocked_samples)
                true_error = (count_result + unblocked_positives - gt) / gt
                # get estimated confidence interval
                if count_result != 0:
                    _, gaussian_upper = get_ci_gaussian(results, config.confidence_level)
                    gaussian_upper *= len(blocked_samples)
                    gaussian_upper_error = (gaussian_upper + unblocked_positives - gt) / gt
                    _, ttest_upper = get_ci_ttest(results, config.confidence_level)
                    ttest_upper *= len(blocked_samples)
                    ttest_upper_error = (ttest_upper + unblocked_positives - gt) / gt
                    gaussian_upper_errors.append(gaussian_upper_error)
                    ttest_upper_errors.append(ttest_upper_error)
                    logging.info(f"running with sample size {sample_size} trial {i} count result {count_result + unblocked_positives} true error {true_error} gaussian upper {gaussian_upper} gaussian upper error {gaussian_upper_error} ttest upper {ttest_upper} ttest upper error {ttest_upper_error}")
                else:
                    logging.info(f"running with sample size {sample_size} trial {i} count result {count_result + unblocked_positives} true error {true_error}")
                
                count_results.append(count_result+unblocked_positives)
                true_errors.append(true_error)

            average_count_result = np.mean(count_results)
            average_true_error = np.mean(true_errors)
            std_true_error = np.std(true_errors)
            gaussian_upper_error = np.mean(gaussian_upper_errors)
            ttest_upper_error = np.mean(ttest_upper_errors)
            logging.info(f"results: sample size {sample_size} average count result {average_count_result} average true error {average_true_error} std true error {std_true_error} gaussian upper error {gaussian_upper_error} ttest upper error {ttest_upper_error}")


