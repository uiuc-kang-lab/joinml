from joinml.proxy.get_proxy import get_proxy
from joinml.dataset_loader import JoinDataset
from joinml.oracle import Oracle
from joinml.config import Config
from joinml.utils import set_up_logging, get_ci_bootstrap, preprocess, normalize, defensive_mix

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

    strata = []
    n = int(len(dataset_sizes[0])*len(dataset_sizes[1]) - config.oracle_budget)
    m = int(config.oracle_budget / (config.num_strata - 1))
    strata.append((0, n))
    allocation_step = m
    for i in range(1, config.num_strata):
        strata.append((strata[i-1][1], strata[i-1][1] + allocation_step))
    logging.info(f"Strata: {strata}")
    strata_allocation = []
    stage_one_sample_size = int(0.9 * config.oracle_budget)
    for stratum in strata:
        prosy_score_sum = np.sum(proxy_scores[proxy_rank[stratum[0]:stratum[1]]]) / (stratum[1] - stratum[0])
        strata_allocation.append(int(stage_one_sample_size * prosy_score_sum / np.sum(proxy_scores)))
    # sample for each stratum
    strata_samples = []
    strata_agg_results = []
    strata_variance = []
    strata_mean = []
    for stratum, allocation in zip(strata, strata_allocation):
        sample_weights = proxy_scores[proxy_rank[stratum[0]:stratum[1]]]
        sample_weights = normalize(sample_weights)
        sample = np.random.choice(stratum[1] - stratum[0], allocation, replace=False, p=sample_weights)
        sample_ids = proxy_rank[stratum[0]:stratum[1]][sample]
        strata_samples.append(sample_ids)
        sample_ids = np.array(np.unravel_index(sample_ids, dataset_sizes)).T
        results = []
        for s_id, s in zip(sample_ids, sample):
            if oracle.query(s_id):
                results.append(1. / len(sample_weights) / sample_weights[s])
            else:
                results.append(0)
        strata_agg_results.append(np.array(results))
        strata_variance.append(np.var(results))
        strata_mean.append(np.mean(results))
    
    # find the optimal sets of stratum samples
    allocation_variances = []
    allocation_means = []
    allocation_utility = []
    for i in range(1, len(strata)):
        # The Eve's law
        total_variance = (n + (i-1)*m) / (n+i*m) * allocation_variances[i-1] + m / (n+i*m) * strata_variance[i] + \
                         m*(n + (i-1)*m) / (n+i*m)**2 * (strata_mean[i] - allocation_means[i-1])**2
        total_mean = (n + (i-1)*m) / (n+i*m) * allocation_means[i-1] + m / (n+i*m) * strata_mean[i]
        allocation_variances.append(total_variance)
        allocation_means.append(total_mean)
        allocation_utility.append(total_variance / i)
    
    # find the optimal allocation
    optimal_allocation = np.argmin(allocation_utility) + 1
    allocation_sample = []
    for i in range(optimal_allocation):
        allocation_sample.append(strata_samples[i])
    allocation_sample = np.concatenate(allocation_sample)
    logging.info(f"Optimal allocation: {optimal_allocation}")
    
    # run bootstrap to get an CI
    allocation_sample_ids = np.array(np.unravel_index(allocation_sample, dataset_sizes)).T
    allocation_results = []
    for s_id in allocation_sample_ids:
        if oracle.query(s_id):
            allocation_results.append(1)
        else:
            allocation_results.append(0)
    allocation_results = np.array(allocation_results)
    ci_lower, ci_upper = get_ci_bootstrap(allocation_results, config.confidence_level, config.bootstrap_trials)
    count_upper = ci_upper * len(allocation_sample)
    
    # run oracle on the remaning data
    blocking_data = []
    for i in range(optimal_allocation, len(strata)):
        blocking_data.append(strata[i])
    blocking_data = np.concatenate(blocking_data)
    logging.info(f"Blocking data size: {len(blocking_data)}")
    blocking_data_ids = np.array(np.unravel_index(blocking_data, dataset_sizes)).T
    blocking_results  = 0
    for s_id in blocking_data_ids:
        if oracle.query(s_id):
            blocking_results += 1
    logging.info(f"Blocking results: {blocking_results}")
    count_upper += blocking_results
    logging.info(f"Result: count upper {count_upper} error rate {abs(count_upper - gt) / gt}")

    
    
    



    





    