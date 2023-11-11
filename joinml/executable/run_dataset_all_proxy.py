from joinml.config import Config
from joinml.commons import dataset2modality, modality2proxy
from joinml.proxy.get_proxy import get_proxy
from joinml.utils import set_random_seed, set_up_logging, normalize
from joinml.dataset_loader import JoinDataset
from joinml.oracle import Oracle
from joinml.meter import ErrorMeter
from joinml.join.run_join import run as run_join

import logging


def run(config: Config):
    # get available proxy
    available_proxy = modality2proxy[dataset2modality[config.dataset_name]]
    # temporary TODO
    available_proxy = available_proxy[available_proxy.index('TF/IDF')+1:]
    # set up
    set_up_logging(config.log_path)
    set_random_seed(config.seed)
    # dataset
    dataset = JoinDataset(config)
    dataset_ids = dataset.get_ids()
    dataset_join_cols = dataset.get_join_column_per_table(dataset_ids)
    # oracle
    oracle = Oracle(config)
    # error meter
    error_meter = ErrorMeter(dataset, oracle, config)
    # run
    for proxy_name in available_proxy:
        try:
            config.proxy = proxy_name
            logging.info(f"Running {proxy_name}...")
            # get proxy
            proxy = get_proxy(config)
            # run proxy
            if config.is_self_join:
                scores = proxy.get_proxy_score_for_tables(dataset_join_cols[0], dataset_join_cols[0])
                scores = [normalize(scores, is_self_join=True)]
            else:
                scores = []
                for i in range(1,len(dataset_join_cols)):
                    score = proxy.get_proxy_score_for_tables(dataset_join_cols[i-1], dataset_join_cols[i])
                    scores.append(normalize(score))
                assert len(scores) == len(dataset_join_cols) - 1
            # run join
            results = run_join(config, scores, dataset_ids, oracle)
        except Exception as e:
            logging.error(f"Error running {proxy_name}: {e}")
            continue
        
        # report
        for result in results:
            error_meter.add_results(result)
        error_meter.report()
        error_meter.reset()



