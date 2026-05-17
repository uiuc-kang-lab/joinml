from joinml.config import Config
from joinml.dataset_loader import JoinDataset
from joinml.commons import kind2proxy
from joinml.utils import preprocess


import logging
import numpy as np
import os
from typing import Union

def get_proxy(config: Config):
    if config.proxy in kind2proxy["image_embedding"]:
        from joinml.proxy.image_embedding_proxy import ImageEmbeddingProxy
        
        return ImageEmbeddingProxy(config)
    elif config.proxy in kind2proxy["text_embedding"]:
        from joinml.proxy.text_embedding_proxy import TextEmbeddingProxy

        return TextEmbeddingProxy(config)
    elif config.proxy in kind2proxy["string_matching"]:
        from joinml.proxy.stringmatching_proxy import StringMatchingProxy
        
        return StringMatchingProxy(config)
    elif config.proxy in kind2proxy["opencv"]:
        from joinml.proxy.opencv_proxy import OpencvProxy

        return OpencvProxy(config)
    elif config.proxy in kind2proxy["multimodal"]:
        from joinml.proxy.multimodal_embedding_proxy import MultiModalEmbeddingProxy

        return MultiModalEmbeddingProxy(config)
    elif config.proxy.startswith("data/") or config.proxy == "":
        from joinml.proxy.text_embedding_proxy import TextEmbeddingProxy
        
        return TextEmbeddingProxy(config)
    else:
        raise NotImplementedError(f"Proxy {config.proxy} not implemented.")

def get_proxy_score(config: Config, dataset: JoinDataset, is_wanderjoin: bool=False) -> np.ndarray:
    # check cache for proxy scores
    if config.proxy == "" and not config.join_reorder:
        proxy_store_path = f"{config.cache_path}/{config.dataset_name}.npy"
    elif config.proxy == "":
        proxy_store_path =  f"{config.cache_path}/{config.dataset_name}_{config.table_ids[0]}{config.table_ids[1]}_scores.npy"
        print(proxy_store_path)
    else:
        proxy_store_path = f"{config.cache_path}/{config.dataset_name}_{config.proxy.split('/')[-1]}_scores.npy"
    dataset_sizes = dataset.get_sizes()
    if config.is_self_join:
        dataset_sizes = [dataset_sizes[0], dataset_sizes[0]]
    if config.proxy_score_cache and os.path.exists(proxy_store_path):
        logging.info("Loading proxy scores from %s", proxy_store_path)
        proxy_scores = np.load(proxy_store_path)
        print(proxy_scores.shape)
        assert np.prod(proxy_scores.shape) == np.prod(dataset_sizes), f"Proxy scores shape does not match dataset sizes: {np.prod(proxy_scores.shape)} != {np.prod(dataset_sizes)}"
        assert isinstance(proxy_scores, np.ndarray), "Proxy scores is not a numpy array."
    else:
        logging.info("Calculating proxy scores.")
        proxy = get_proxy(config)
        join_columns = dataset.get_join_column()
        if config.is_self_join:
            join_columns = [join_columns[0], join_columns[0]]

        # FIXME: special case of three tables
        if config.dataset_name == "city_human_three":
            logging.info("calculating proxy scores for table 0, 1")
            proxy_scores1 = proxy.get_proxy_score_for_tables(join_columns[0], join_columns[1])
            logging.info("calculating proxy scores for table 1, 2")
            proxy_scores2 = proxy.get_proxy_score_for_tables(join_columns[1], join_columns[2])
            np.save(f"{config.cache_path}/city_human_three_01_scores.npy", proxy_scores1)
            np.save(f"{config.cache_path}/city_human_three_12_scores.npy", proxy_scores2)
            assert proxy_scores1.shape[1] == proxy_scores2.shape[0]
            proxy_scores1 = preprocess(proxy_scores1)
            proxy_scores2 = preprocess(proxy_scores2)
            proxy_scores = np.einsum('ij,jk->ijk', proxy_scores1, proxy_scores2)
            proxy_scores = preprocess(proxy_scores)
            proxy_scores = proxy_scores.flatten()
            np.save(f"{proxy_store_path}", proxy_scores)
            return proxy_scores
        elif config.dataset_name == "quora_three":
            logging.info("calculating proxy scores for table 0, 1")
            proxy_scores1 = proxy.get_proxy_score_for_tables(join_columns[0], join_columns[1])
            logging.info("calculating proxy scores for table 1, 2")
            proxy_scores2 = proxy.get_proxy_score_for_tables(join_columns[1], join_columns[2])
            np.save(f"{config.cache_path}/quora_three_01_scores.npy", proxy_scores1)
            np.save(f"{config.cache_path}/quora_three_12_scores.npy", proxy_scores2)
            assert proxy_scores1.shape[1] == proxy_scores2.shape[0]
            proxy_scores1 = preprocess(proxy_scores1)
            proxy_scores2 = preprocess(proxy_scores2)
            proxy_scores = np.einsum('ij,jk->ijk', proxy_scores1, proxy_scores2)
            proxy_scores = preprocess(proxy_scores)
            proxy_scores = proxy_scores.flatten()
            np.save(f"{proxy_store_path}", proxy_scores)
            return proxy_scores

        proxy_scores = proxy.get_proxy_score_for_tables(join_columns[0], join_columns[1])
        logging.info(f"Postprocessing proxy scores of shape {proxy_scores.shape} VS {len(join_columns[0])}|{len(join_columns[1])}.")
        proxy_scores = preprocess(proxy_scores, is_self_join=config.is_self_join)
        proxy_scores = proxy_scores.flatten()

        if config.proxy_score_cache:
            logging.info("Saving proxy scores to %s", proxy_store_path)
            np.save(proxy_store_path, proxy_scores)

    if is_wanderjoin:
        proxy_scores = np.reshape(proxy_scores, dataset_sizes)
        proxy_scores = proxy_scores / np.sum(proxy_scores, axis=1).reshape(-1, 1)
        proxy_scores = proxy_scores / proxy_scores.shape[0]
        proxy_scores = proxy_scores.flatten()

    return proxy_scores

def get_proxy_rank(config: Config, dataset: JoinDataset, proxy_scores: Union[np.ndarray,None]=None):
    dataset_sizes = dataset.get_sizes()
    if config.is_self_join:
        dataset_sizes = [dataset_sizes[0], dataset_sizes[0]]
    if proxy_scores is None:
        proxy_scores = get_proxy_score(config, dataset)
    if config.proxy == "" and not config.join_reorder:
        proxy_rank_store_path = f"{config.cache_path}/{config.dataset_name.replace('scores', 'rank')}.npy"
    elif config.proxy == "":
        proxy_rank_store_path = f"{config.cache_path}/{config.dataset_name}_{config.table_ids[0]}{config.table_ids[1]}_rank.npy"
    else:
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