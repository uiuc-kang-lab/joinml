from joinml.config import Config
from joinml.utils import set_up_logging
from joinml.proxy.get_proxy import get_proxy_score, get_proxy_rank
from joinml.dataset_loader import load_dataset
from joinml.oracle import Oracle

import logging
import numpy as np

def run(config: Config):
    set_up_logging(config.log_path, config.log_level)

    # log config
    logging.info(config)

    # dataset, oracle
    dataset = load_dataset(config)
    oracle = Oracle(config)

    # setup dataset
    dataset_sizes = dataset.get_sizes()
    if config.is_self_join:
        dataset_sizes = (dataset_sizes[0], dataset_sizes[0])
    
    count_gt, sum_gt, avg_gt = dataset.get_gt(oracle)

    logging.debug(f"count gt: {count_gt}, sum gt: {sum_gt}, avg gt: {avg_gt}")

    # get proxy
    proxy_scores = get_proxy_score(config, dataset)
    proxy_rank = get_proxy_rank(config, dataset, proxy_scores)

    # divide into k buckets based on proxy rank
    n_bucket = 100
    bucket_size = len(proxy_rank) // n_bucket
    buckets = []
    for i in range(n_bucket):
        if i != n_bucket - 1:
            buckets.append(proxy_rank[i * bucket_size: (i + 1) * bucket_size])
        else:
            buckets.append(proxy_rank[i * bucket_size:])
    
    # calculate the final variance of each bucket
    variance = []
    for i, bucket in enumerate(buckets):
        bucket_proxy_score = proxy_scores[bucket]
        # normalize the proxy score
        bucket_proxy_score = bucket_proxy_score / np.sum(bucket_proxy_score)
        results = []
        bucket_results = []

        bucket_pairs = np.array(np.unravel_index(bucket, dataset_sizes)).T
        assert len(bucket_pairs) == len(bucket_proxy_score)
        N = len(bucket_proxy_score)
        
        for data_pair, proxy_score in zip(bucket_pairs, bucket_proxy_score):
            if oracle.query(data_pair):
                stats = dataset.get_statistics(data_pair)
                result = stats**2 * 1/(proxy_score * N)
                results.append(result)
                bucket_results.append(stats)
            else:
                results.append(0)
                bucket_results.append(0)
        
        var = np.mean(results) - np.mean(bucket_results)**2
        logging.info(f"variance at bucket {i}: {var}")
        variance.append(var)


    
    

