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
        proxy_rank = np.array(proxy_rank, dtype=np.intp)
        assert np.prod(proxy_rank.shape) == np.prod(dataset_sizes), "Proxy rank shape does not match dataset sizes."
    else:
        logging.info("Calculating proxy rank.")
        proxy_rank = np.argsort(proxy_scores)
        if config.blocking_cache:
            logging.info("Saving proxy rank to %s", proxy_rank_store_path)
            np.save(proxy_rank_store_path, proxy_rank)

    # divide population into strata
    strata = [(0, proxy_rank.shape[0] - int(config.oracle_budget*0.2))]
    sample_size = int(0.8 * config.oracle_budget)
    blocking_size_upperbound = config.oracle_budget - sample_size
    blocking_stratum_size = int(blocking_size_upperbound / (config.num_strata - 1))
    for i in range(config.num_strata-1):
        if i < config.num_strata - 2:
            strata.append((strata[i][1], strata[i][1]+blocking_stratum_size))
        else:
            strata.append((strata[i][1], proxy_rank.shape[0]))

    # get groundtruth for each stratum
    strata_gts = []
    for i in range(1, config.num_strata):
        data_ids = proxy_rank[strata[i][0]: strata[i][1]]
        data_ids = np.array(np.unravel_index(data_ids, dataset_sizes)).T
        count = 0
        for data_id in data_ids:
            if oracle.query(data_id):
                count += 1
        if config.debug:
            logging.info(f"strata {i}: {strata[i]}, #positive: {count}")
        strata_gts.append(count)
    strata_gts = [gt - sum(strata_gts)] + strata_gts
    if config.debug:
        logging.info(f"strata {0}: {strata[0]}, #positive: {strata_gts[0]}")

    # proxy weights for each stratum
    strata_proxy_scores = []
    strata_sample_sizes = []
    strata_population = []
    total_proxy_score = np.sum(proxy_scores)
    for (stratum_begin, stratum_end) in strata:
        stratum_population = proxy_rank[stratum_begin:stratum_end]
        strata_population.append(stratum_population)
        stratum_proxy_scores = proxy_scores[stratum_population]
        total_stratum_proxy_score = np.sum(stratum_proxy_scores)
        stratum_sample_size = int(sample_size * total_stratum_proxy_score / total_proxy_score)
        strata_proxy_scores.append(stratum_proxy_scores)
        strata_sample_sizes.append(stratum_sample_size)
        if config.debug:
            logging.info(f"strata sample size: {stratum_sample_size}")
    
    # sample for each strata
    strata_samples = []
    # strata_samples_id = []
    # strata_sample_positiveness = []
    # strata_proxy_weights = []
    for i, (stratum_begin, stratum_end) in enumerate(strata):
        stratum_proxy_scores = strata_proxy_scores[i]
        stratum_proxy_weights = normalize(stratum_proxy_scores)
        assert abs(np.sum(stratum_proxy_weights) - 1) < 1e-3
        stratum_sample = np.random.choice(a=len(stratum_proxy_weights), size=strata_sample_sizes[i], p=stratum_proxy_weights, replace=True)
        stratum_sample_id = strata_population[i][stratum_sample]
        # strata_samples.append(stratum_sample)
        stratum_sample_id = np.array(np.unravel_index(stratum_sample_id, dataset_sizes)).T
        # strata_samples_id.append(stratum_sample_id)
        stratum_sample_positiveness = []
        results = []
        for stratum_s_id, stratum_s in zip(stratum_sample_id, stratum_sample):
            if oracle.query(stratum_s_id):
                stratum_sample_positiveness.append(1)
                results.append(1 / len(stratum_proxy_weights) / stratum_proxy_weights[stratum_s])
            else:
                stratum_sample_positiveness.append(0)
                results.append(0)
        results = np.array(results)
        stratum_sample_mean = results.mean()
        stratum_sample_var = results.var()
        stratum_count = stratum_sample_mean * len(stratum_proxy_weights)
        if config.debug:
            logging.info(f"stratum {i}: mean {stratum_sample_mean} var {stratum_sample_var} count {stratum_count}")

        stratum_sample_positiveness = np.array(stratum_sample_positiveness)
        # strata_sample_positiveness.append(stratum_sample_positiveness)
        # strata_proxy_weights.append(stratum_proxy_weights)
        strata_samples.append(results)
    
    # bootstrap trials
    bootstrap_results = []
    for trials in range(config.bootstrap_trials):
        # resample
        strata_resamples_means = []
        strata_resamples_vars = []
        for i in range(len(strata)):
            # strata_resample_weights = strata_proxy_scores[i][strata_samples[i]]
            # strata_resample_weights = normalize(strata_resample_weights)
            # stratum_resample = np.random.choice(len(strata_samples[i]), len(strata_samples[i]), p=strata_resample_weights, replace=True)
            # stratum_sample_proxy_scores = strata_proxy_scores[i][strata_samples[i]]
            # stratum_resample_id = strata_samples_id[i][stratum_resample]
            # stratum_resample_positiveness = strata_sample_positiveness[i][stratum_resample]
            # stratum_resample_proxy_weights = strata_proxy_weights[i][stratum_resample]
            # results = []
            # for positiveness, proxy_weight in zip(stratum_resample_positiveness, stratum_resample_proxy_weights):
            #     if positiveness == 1:
            #         results.append(1 / len(strata_population[i]) / proxy_weight)
            #     else:
            #         results.append(0)
            # results = np.array(results)
            results = np.random.choice(strata_samples[i], size=len(strata_samples[i]), replace=True)
            stratum_resample_mean = results.mean()
            stratum_resample_var = results.var()
            stratum_resample_count = stratum_resample_mean * len(strata_samples[i])
            strata_resamples_means.append(stratum_resample_mean)
            strata_resamples_vars.append(stratum_resample_var)
            if config.debug:
                logging.info(f"stratum {i} resample: mean {stratum_resample_mean} variance {stratum_resample_var} count {stratum_resample_count}")

        # calculate the variance of subpopulations
        subpopulation_variances = []
        for i in range(1, len(strata)):
            all_strata_sizes = sum([len(stratum_population) for stratum_population in strata_population[:i+1]])
            variance = 0
            for j in range(i+1):
                stratum_population_size = len(strata_population[j])
                stratum_variance = (stratum_population_size / all_strata_sizes)**2 * \
                                   strata_resamples_vars[j]**2  / strata_sample_sizes[j] 
                variance += stratum_variance
            subpopulation_variances.append(variance)
        subpopulation_variances = np.array(subpopulation_variances)
        # get the minimum variance
        min_var = np.min(subpopulation_variances)
        optimal_allocation = 1 + [i for i in range(subpopulation_variances.shape[0]) if subpopulation_variances[i] == min_var][0]
        if config.debug:
            logging.info(f"optimal allocation {optimal_allocation}")
        # calculate the statistics of the optimal allocation strata
        sampling_mean = 0
        sampling_population_size = sum([len(stratum_population) for stratum_population in strata_population[:optimal_allocation+1]])
        for i in range(optimal_allocation+1):
            mean = strata_resamples_means[i] * len(strata_population[i]) / sampling_population_size
            sampling_mean += mean
        sampling_count = sampling_mean * sampling_population_size
        # calculate the statistics of the blocking strata
        blocking_count = 0
        for i in range(optimal_allocation+1, config.num_strata):
            blocking_count += strata_gts[i]
        
        total_count = blocking_count + sampling_count
        bootstrap_results.append(total_count)
        error = (total_count - gt) / gt
        if config.debug:
            logging.info(f"sampling count {sampling_count} blocking count {blocking_count} total_count {total_count} error {error}")
    bootstrap_results = np.sort(bootstrap_results)
    mean = np.mean(bootstrap_results)
    upperbound_idx = int((config.confidence_level + (1-config.confidence_level)/2) * len(bootstrap_results))
    lowerbound_idx = int((1- config.confidence_level) / 2 * len(bootstrap_results))
    upperbound = bootstrap_results[upperbound_idx]
    lowerbound = bootstrap_results[lowerbound_idx]
    logging.info(f"result {mean} with CI {lowerbound} {upperbound}")






