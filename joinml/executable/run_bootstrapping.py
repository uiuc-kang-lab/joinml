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
    n = int(dataset_sizes[0]*dataset_sizes[1] - config.oracle_budget)
    m = int(config.oracle_budget / (config.num_strata - 1))
    strata.append((0, n))
    strata_gts = []
    allocation_step = m
    for i in range(1, config.num_strata):
        strata.append((strata[i-1][1], strata[i-1][1] + allocation_step))
        strata_gt = 0
        for j in range(strata[i][0], strata[i][1]):
            table_id = np.array(np.unravel_index(proxy_rank[j], dataset_sizes)).T
            if oracle.query(table_id):
                strata_gt += 1
        strata_gts.append(strata_gt)
    logging.info(f"Strata: {strata}")
    logging.info(f"Strata gts: {strata_gts}")
    strata_allocation = []
    stage_one_sample_size = int(0.9 * config.oracle_budget)
    for stratum in strata:
        prosy_score_sum = np.sum(proxy_scores[proxy_rank[stratum[0]:stratum[1]]])
        strata_allocation.append(int(stage_one_sample_size * prosy_score_sum / np.sum(proxy_scores)))
    # sample for each stratum
    strata_samples = []
    strata_agg_results = []
    strata_variance = []
    strata_mean = []
    for i, (stratum, allocation) in enumerate(zip(strata, strata_allocation)):
        sample_weights = proxy_scores[proxy_rank[stratum[0]:stratum[1]]]
        sample_weights = normalize(sample_weights)
        sample = np.random.choice(stratum[1] - stratum[0], allocation, replace=True, p=sample_weights)
        logging.info(f"Strata {i} sample size: {len(sample)}")
        sample_ids = proxy_rank[stratum[0]:stratum[1]][sample]
        strata_samples.append(sample_ids)
        sample_ids = np.array(np.unravel_index(sample_ids, dataset_sizes)).T
        results = []
        for s_id, s in zip(sample_ids, sample):
            if oracle.query(s_id):
                results.append(1. / len(sample_weights) / sample_weights[s])
            else:
                results.append(0)
        logging.info(f"Strata {i} results: {results}")
        strata_agg_results.append(np.array(results))
        strata_variance.append(np.var(results))
        strata_mean.append(np.mean(results))
        logging.info(f"Strata {i} variance {strata_variance[i]} mean {strata_mean[i]}")
    
    for i in range(config.bootstrap_trials):
        # resample each stratum
        strata_resamples = []
        for j in range(0, len(strata)):
            strata_resample = np.random.choice(len(strata_agg_results[j]), len(strata_agg_results[j]), replace=True)
            strata_resample = strata_agg_results[j][strata_resample]
            strata_resamples.append(strata_resample)
        # get a allocation fraction
        errors = []
        for j in range(1, len(strata_resamples)):
            resamples = [strata_resamples[k] for k in range(j+1)]
            resamples = np.concatenate(resamples)
            errors.append(np.var(resamples) / j)
        errors = np.array(errors)
        allocation_fraction = np.argmin(errors) + 1
        logging.info(f"Bootstrap trial {i} allocation fraction {allocation_fraction}")
        # get the allocation resample
        allocation_resample = [strata_resamples[k] for k in range(allocation_fraction+1)]
        mean = np.mean(np.concatenate(allocation_resample))
        sampling_agg_results = mean * ((allocation_fraction - 1) * m + n)
        blocking_agg_results = 0
        for j in range(allocation_fraction+1, len(strata_resamples)):
            blocking_agg_results += strata_gts[j]
        agg_results = sampling_agg_results + blocking_agg_results
        agg_error = (agg_results - gt) / gt
        logging.info(f"Bootstrap trial {i} agg results {agg_results} agg error {agg_error}")    
