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

    # run variance estimation
    start_rank = len(proxy_rank) - config.oracle_budget
    end_rank = len(proxy_rank)
    logging.info("Running variance estimation.")
    variance_estimation = []
    upper_error_rates_estimation = []
    true_error_rates_estimation = []
    for rank_div in range(start_rank, end_rank, config.allocation_step):
        # get gt
        subpopulation_gt = gt
        for data_id in np.array(np.unravel_index(proxy_rank[rank_div:], dataset_sizes)).T:
            if oracle.query(data_id):
                subpopulation_gt -= 1
        logging.info(f"groundtruth of the subpopulation: {subpopulation_gt}")
        subpopulation_variance = []
        subpopulation_upper_errors = []
        subpopulation_upper_error_rates = []
        subpopulation_true_errors = []
        subpopulation_true_error_rates = []
        subpopulation_proxy_scores = proxy_scores[proxy_rank[:rank_div]]
        subpopulation_proxy_scores = normalize(subpopulation_proxy_scores)
        for i in range(config.repeats):
            sample = np.random.choice(rank_div, size=config.sample_size, p=subpopulation_proxy_scores, replace=True)
            sample_id = proxy_rank[sample]
            sample_id = np.array(np.unravel_index(sample_id, dataset_sizes)).T
            results = []
            for s, s_id in zip(sample, sample_id):
                if oracle.query(s_id):
                    results.append(1. / rank_div / subpopulation_proxy_scores[s])
                else:
                    results.append(0.)

            results = np.array(results)                
            variance = np.var(results)
            _, gaussian_upper = get_ci_gaussian(results, config.confidence_level)
            upper_error = abs(gaussian_upper*rank_div - subpopulation_gt)
            upper_error_rate = upper_error / gt
            true_error = abs(np.mean(results) * rank_div - subpopulation_gt)
            true_error_rate = true_error / gt
            logging.info(f"subpopulation {rank_div} repeat {i} variance {np.var(sample)} upper error {upper_error} upper_error_rate {upper_error_rate} true_error {true_error} true_error_rate {true_error_rate}")
            subpopulation_variance.append(variance)
            subpopulation_upper_errors.append(upper_error)
            subpopulation_upper_error_rates.append(upper_error_rate)
            subpopulation_true_errors.append(true_error)
            subpopulation_true_error_rates.append(true_error_rate)
        variance_estimation.append(np.mean(subpopulation_variance))
        upper_error_rates_estimation.append(np.mean(subpopulation_upper_error_rates))
        true_error_rates_estimation.append(np.mean(subpopulation_true_errors))
        logging.info(f"average: subpopulation {rank_div} variance {np.mean(subpopulation_variance)} upper_error_rate {np.mean(subpopulation_upper_error_rates)} true_error_rate {np.mean(subpopulation_true_error_rates)}")
