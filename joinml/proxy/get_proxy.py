from joinml.proxy.image_embedding_proxy import ImageEmbeddingProxy
from joinml.proxy.stringmatching_proxy import StringMatchingProxy
from joinml.proxy.opencv_proxy import OpencvProxy
from joinml.proxy.text_embedding_proxy import TextEmbeddingProxy
from joinml.proxy.multimodal_embedding_proxy import MultiModalEmbeddingProxy
from joinml.config import Config
from joinml.dataset_loader import JoinDataset
from joinml.commons import kind2proxy
from joinml.utils import preprocess

import logging
import numpy as np
import os

def get_proxy(config: Config):
    if config.proxy in kind2proxy["image_embedding"]:
        return ImageEmbeddingProxy(config)
    elif config.proxy in kind2proxy["text_embedding"]:
        return TextEmbeddingProxy(config)
    elif config.proxy in kind2proxy["string_matching"]:
        return StringMatchingProxy(config)
    elif config.proxy in kind2proxy["opencv"]:
        return OpencvProxy(config)
    elif config.proxy in kind2proxy["multimodal"]:
        return MultiModalEmbeddingProxy(config)
    elif config.proxy.startswith("data/"):
        return TextEmbeddingProxy(config)
    else:
        raise NotImplementedError(f"Proxy {config.proxy} not implemented.")

def get_proxy_score(config: Config, dataset: JoinDataset) -> np.ndarray:
    # check cache for proxy scores
    proxy_store_path = f"{config.cache_path}/{config.dataset_name}_{config.proxy.split('/')[-1]}_scores.npy"
    dataset_sizes = dataset.get_sizes()
    if config.proxy_score_cache and os.path.exists(proxy_store_path):
        logging.info("Loading proxy scores from %s", proxy_store_path)
        proxy_scores = np.load(proxy_store_path)
        assert np.prod(proxy_scores.shape) == np.prod(dataset_sizes), "Proxy scores shape does not match dataset sizes."
        assert isinstance(proxy_scores, np.ndarray), "Proxy scores is not a numpy array."
    else:
        logging.info("Calculating proxy scores.")
        proxy = get_proxy(config)
        join_columns = dataset.get_join_column()
        if config.is_self_join:
            join_columns = [join_columns[0], join_columns[0]]
        proxy_scores = proxy.get_proxy_score_for_tables(join_columns[0], join_columns[1])
        logging.info(f"Postprocessing proxy scores of shape {proxy_scores.shape} VS {len(join_columns[0])}|{len(join_columns[1])}.")
        proxy_scores = preprocess(proxy_scores, is_self_join=config.is_self_join)
        proxy_scores = proxy_scores.flatten()

        if config.proxy_score_cache:
            logging.info("Saving proxy scores to %s", proxy_store_path)
            np.save(proxy_store_path, proxy_scores)

    return proxy_scores

def get_proxy_rank(config: Config, dataset: JoinDataset, proxy_scores: np.ndarray|None=None):
    dataset_sizes = dataset.get_sizes()
    if proxy_scores is None:
        proxy_scores = get_proxy_score(config, dataset)
    proxy_rank_store_path = f"{config.cache_path}/{config.dataset_name}_{config.proxy.split('/')[-1]}_rank.npy"
    if config.proxy_score_cache and os.path.exists(proxy_rank_store_path):
        logging.info("Loading proxy rank from %s", proxy_rank_store_path)
        proxy_rank = np.load(proxy_rank_store_path)
        assert np.prod(proxy_rank.shape) == np.prod(dataset_sizes), "Proxy rank shape does not match dataset sizes."
    else:
        logging.info("Calculating proxy rank.")
        proxy_rank = np.argsort(proxy_scores)
        if config.proxy_score_cache:
            logging.info("Saving proxy rank to %s", proxy_rank_store_path)
            np.save(proxy_rank_store_path, proxy_rank)
    return proxy_rank