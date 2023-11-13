from joinml.proxy.image_embedding_proxy import ImageEmbeddingProxy
from joinml.proxy.stringmatching_proxy import StringMatchingProxy
from joinml.proxy.opencv_proxy import OpencvProxy
from joinml.proxy.text_embedding_proxy import TextEmbeddingProxy
from joinml.config import Config
from joinml.commons import modality2proxy, dataset2modality, kind2proxy
from joinml.dataset_loader import JoinDataset
from joinml.oracle import Oracle
from joinml.proxy.evaluation import Evaluator

import logging

def get_proxy(config: Config):
    if config.proxy in kind2proxy["image_embedding"]:
        return ImageEmbeddingProxy(config)
    elif config.proxy in kind2proxy["text_embedding"]:
        return TextEmbeddingProxy(config)
    elif config.proxy in kind2proxy["string_matching"]:
        return StringMatchingProxy(config)
    elif config.proxy in kind2proxy["opencv"]:
        return OpencvProxy(config)
    elif config.proxy.startswith("data/"):
        return TextEmbeddingProxy(config)
    else:
        raise NotImplementedError(f"Proxy {config.proxy} not implemented.")
    
def get_proxy_by_evaluation(config: Config, dataset: JoinDataset, oracle: Oracle):
    available_proxies = modality2proxy[dataset2modality[config.dataset_name]]
    evaluator = Evaluator(config, dataset)
    eval_results = {}
    for proxy in available_proxies:
        logging.info(f"Evaluating {proxy}...")
        config.proxy = proxy
        try:
            mse = evaluator.evaluate(get_proxy(config), oracle)
            eval_results[proxy] = mse
            logging.info(f"Proxy {proxy} evaluated with MSE {mse}")
        except Exception as e:
            logging.info(f"Error evaluating {proxy}: {e}")
    logging.info(f"Proxy evaluation results: {eval_results}")
    eval_result_list = [(k, v) for k, v in eval_results.items()]
    eval_result_list.sort(key=lambda x: x[1])
    return get_proxy(eval_result_list[0][0])