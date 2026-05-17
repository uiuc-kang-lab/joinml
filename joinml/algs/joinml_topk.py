from joinml.proxy.get_proxy import get_proxy_score, get_proxy_rank
from joinml.scalable_dataset_loader import ScalableJoinDataset, load_dataset
from joinml.oracle import Oracle
from joinml.config import Config
from joinml.utils import set_up_logging, normalize
from joinml.estimates import TopK
from joinml.utils import weighted_sample_pd, get_ci_bootstrap_ttest

import logging
import numpy as np
from typing import Tuple, List

def adaptive_allocation(strata_population: List[int],
                        strata_count_variance: List[float], strata_count_mean: List[float]
                        ) -> Tuple[List[int], List[int]]:
    # calculate the total variance of the sub population
    count_total_variances = []
    for i in range(len(strata_population)):
        count_variance = 0
        count_mean = 0
        total_population = sum([strata_population[j] for j in range(i+1)])
        for j in range(i+1):
            count_variance += (strata_population[j] / total_population)**2 * strata_count_variance[j]
            count_mean += (strata_population[j] / total_population) * strata_count_mean[j]
        count_total_variances.append(count_variance)


    print(f"count variances: {count_total_variances}")

    # find the best allocation according to aggregator
    min_variance = np.min(count_total_variances)
    optimal_allocation = 1 + [i for i in range(len(count_total_variances)) if count_total_variances[i] == min_variance][-1]
    logging.debug(f"optimal allocation {optimal_allocation} utility {min_variance}")
    strata_for_sampling = list(range(optimal_allocation))
    strata_for_blocking = list(range(optimal_allocation, len(strata_population)))
    return strata_for_sampling, strata_for_blocking

def run_once(config: Config, dataset: ScalableJoinDataset, count_gt, strata_sample_sizes):
    strata_sample_count_results = []    # O(x)
    strata_count_mean = []       # mean of O(x)
    strata_count_mean_vars = []         # variance of O(x)
    # run sampling with replacement for each stratum
    for stratum_id in range(dataset.K+1):
        stratum_sample_count_results = dataset.sample(stratum_id, strata_sample_sizes[stratum_id], replace=True)
        stratum_count_mean = {fabric: np.mean(stratum_sample_count_results[fabric]) for fabric in stratum_sample_count_results}
        stratum_count_mean_var = {fabric: np.var(stratum_sample_count_results[fabric], ddof=1).item() / len(stratum_sample_count_results[fabric]) for fabric in stratum_sample_count_results}
        strata_sample_count_results.append(stratum_sample_count_results)
        strata_count_mean.append(stratum_count_mean)
        strata_count_mean_vars.append(stratum_count_mean_var)

    print(f"strata count vars: {strata_count_mean_vars}")
    print(f"strata count means: {strata_count_mean}")

    strata_population_size = [dataset.get_stratum_size(i) for i in range(dataset.K+1)]

    # if sum(strata_sample_count_results[0]) == 0:
    #     sampling_strata = [0]
    #     blocking_strata = [i for i in range(1, dataset.K+1)]
    # else:
    sampling_strata = []
    blocking_strata = []
    for fabric in strata_count_mean[0].keys():
        strata_count_mean_fabric = [strata_count_mean[i][fabric] for i in range(len(strata_count_mean))]
        strata_count_mean_var_fabric = [strata_count_mean_vars[i][fabric] for i in range(len(strata_count_mean_vars))]
        sampling_strata_fabric, blocking_strata_fabric = adaptive_allocation(strata_population_size,
                                                               strata_count_mean_var_fabric, strata_count_mean_fabric)
        if len(blocking_strata_fabric) > len(blocking_strata):
            blocking_strata = blocking_strata_fabric
            sampling_strata = sampling_strata_fabric

    print(f"{config.aggregator} sampling strata: {sampling_strata}, {config.aggregator} blocking strata: {blocking_strata}")

    # allocate extra sample size
    reserved_blocking_cost = np.sum([dataset.get_stratum_size(i) for i in range(1,dataset.K+1)])
    total_extra_sample_size = reserved_blocking_cost - np.sum([strata_population_size[i] for i in blocking_strata])
    print(f"extra sample size after blocking decision, {config.aggregator}: {total_extra_sample_size}")

    # sample size allocation
    original_count_sample_size = np.array([strata_sample_sizes[i] for i in sampling_strata])
    extra_sample_size = np.array(total_extra_sample_size * original_count_sample_size / np.sum(original_count_sample_size), dtype=int)
    print(f"extra sample size after blocking decision per sampling strata, {config.aggregator}: {extra_sample_size}")

    # sample more
    if np.sum(extra_sample_size) > 0:
        for strata_id, extra_size in zip(sampling_strata, extra_sample_size):
            if extra_size == 0:
                continue
            extra_stratum_sample_count_results = dataset.sample(strata_id, extra_size, replace=True)
            for fabric in strata_sample_count_results[strata_id]:
                if fabric in extra_stratum_sample_count_results:
                    strata_sample_count_results[strata_id][fabric] = np.concatenate(
                        (strata_sample_count_results[strata_id][fabric],
                         np.array(extra_stratum_sample_count_results[fabric]))
                    )
        # print("after sampling more, strata count means: {}".format([np.mean(stratum_sample_count_result) for stratum_sample_count_result in strata_sample_count_results]))
        # print("after sampling more, strata count vars: {}".format([np.var(stratum_sample_count_result, ddof=1).item() / len(stratum_sample_count_result) for stratum_sample_count_result in strata_sample_count_results]))

    # run sampling with replacement for blocking strata
    if len(blocking_strata) > 0:
        for i in blocking_strata:
            blocking_ids = dataset.strata[i-1][[f"table_{i}" for i in range(len(dataset.tables))]].values
            blocking_sample_ids = np.random.choice(len(blocking_ids), len(blocking_ids), replace=True)
            blocking_sample_ids = np.array(blocking_ids)[blocking_sample_ids]
            for blocking_s_id in blocking_sample_ids:
                fabrics = dataset.run_oracle(blocking_s_id)
                for fabric in dataset.all_fabrics:
                    if fabric in fabrics:
                        strata_sample_count_results[i][fabric].append(1)
                    else:
                        strata_sample_count_results[i][fabric].append(0)
        # print("after blocking more, strata count means: {}".format([np.mean(stratum_sample_count_result) for stratum_sample_count_result in strata_sample_count_results]))
        # print("after blocking more, strata count vars: {}".format([np.var(stratum_sample_count_result, ddof=1).item() / len(stratum_sample_count_result) for stratum_sample_count_result in strata_sample_count_results]))

    # vectorize the sample results
    # strata_sample_count_results = [np.array(stratum_sample_count_result) for stratum_sample_count_result in strata_sample_count_results]

    # debug info
    print(f"total sample size {[len(stratum_sample_result) for stratum_sample_result in strata_sample_count_results]}")

    # get the estimation
    count_estimation = {}
    for stratum_id, stratum_sample_count_result in enumerate(strata_sample_count_results):
        for fabric in stratum_sample_count_result:
            if fabric not in count_estimation:
                count_estimation[fabric] = 0
            count_estimation[fabric] += np.mean(stratum_sample_count_result[fabric]) * dataset.get_stratum_size(stratum_id)

    logging.debug(f"count estimation {count_estimation}")

    estimate_topk = [item[0] for item in sorted(count_estimation.items(), key=lambda x: x[1], reverse=True)[:config.top_k]]
    gt = count_gt

    print(f"point estimate: {estimate_topk}, gt: {gt}")
    est = TopK(config.oracle_budget, set(gt), set(estimate_topk))
    est.log()
    est.save(config.output_file, f"_topk")
    return

def stats_func(strata_population, strata_sample_count_results, strata_sample_sum_results) -> Tuple[float, float, float]:
    population_size = sum([len(stratum_population) for stratum_population in strata_population])
    sample_size = sum([len(stratum_sample_result) for stratum_sample_result in strata_sample_count_results])
    count_estimation, sum_estimation, avg_estimation = get_estimation(strata_population, strata_sample_count_results, strata_sample_sum_results)
    count_var = get_stratified_var(strata_population, strata_sample_count_results)
    avg_estimation -= get_avg_correction(population_size, sample_size, count_estimation, avg_estimation, count_var)
    return count_estimation, sum_estimation, avg_estimation

def get_estimation(strata_population, strata_sample_count_results, strata_sample_sum_results):
    count_estimation = 0
    sum_estimation = 0
    avg_estimation = 0
    for stratum_sample_count_result, stratum_sample_sum_result, stratum_population in zip(strata_sample_count_results, strata_sample_sum_results, strata_population):
        count_estimation += np.mean(stratum_sample_count_result) * len(stratum_population)
        sum_estimation += np.mean(stratum_sample_sum_result) * len(stratum_population)
    avg_estimation = sum_estimation / count_estimation
    return count_estimation, sum_estimation, avg_estimation

def get_avg_correction(population_size, sample_size, count_mean, avg_mean, count_var):
    return (population_size-sample_size)/(population_size-1) * avg_mean * count_var / sample_size / count_mean**2

def get_stratified_var(strata_population, strata_sample_results):
    variance = 0
    population_size = sum([len(stratum_population) for stratum_population in strata_population])
    for stratum_population, stratum_sample_result in zip(strata_population, strata_sample_results):
        sample_size = len(stratum_sample_result)
        stratum_size = len(stratum_population)
        stratum_var = np.var(stratum_sample_result, ddof=1).item()
        variance += (stratum_size / population_size)**2 * stratum_var / sample_size
    variance *= population_size**2
    return variance

def get_avg_var(sample_size, count_var, count_mean, sum_var, sum_mean):
    return 1. / sample_size * (sum_var / count_mean**2 + count_var * sum_mean**2 / count_mean**4)

def run(config: Config):
    set_up_logging(config.log_path, config.log_level)

    # log config
    logging.info(config)

    # dataset, oracle
    dataset = load_dataset(config)
    count_gt= dataset.get_gt(config.top_k)
    print(f"topk gt: {count_gt}")

    # divide population into strata:
    # 1. a big stratum to only run sampling algorithm
    # 2. a series of strata to potentially run blocking algorithm
    sample_size = int((1-config.max_blocking_ratio) * config.oracle_budget)
    blocking_size_upperbound = config.oracle_budget - sample_size
    dataset.stratify(blocking_size_upperbound)
    strata_sample_size = [1000 for _ in range(dataset.K+1)]
    strata_sample_size[0] = int(sample_size * (1 - dataset.total_blocking_scores))

    print(blocking_size_upperbound)
    print(strata_sample_size)

    for exp_id in range(config.internal_loop):
        print(f"running {exp_id} experiments")
        run_once(config, dataset, count_gt, strata_sample_size)
