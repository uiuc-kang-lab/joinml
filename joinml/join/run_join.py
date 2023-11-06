from joinml.join.naive_importance import run_naive_importance
from joinml.join.naive_uniform import run_uniform
from joinml.join.weighted_wander import run_weighted_wander
from joinml.join.ripple import run_ripple
from joinml.config import Config
from joinml.oracle import Oracle
from typing import List
import numpy as np
from numba import njit

def _preprocess_scores(scores: List[np.ndarray]) -> np.ndarray:
    # expand dimension for each scores
    all_scores = None
    for i in range(len(scores)):
        expanded_axies = []
        if i-1 >= 0:
            expanded_axies += list(range(0,i))
        if i+2 < len(scores)+1:
            expanded_axies += list(range(i+2, len(scores)+1))
        if i == 0:
            all_scores = np.expand_dims(scores[i], axis=tuple(expanded_axies))
        else:
            all_scores = all_scores * np.expand_dims(scores[i], axis=tuple(expanded_axies))
        assert len(all_scores.shape) == len(scores) + 1
    all_scores /= scores[0].shape[0]
    return all_scores

        
    
def run(config, scores: List[np.ndarray], ids: List[List[int]], oracle: Oracle):
    if config.is_self_join:
        assert len(ids) == 1
        ids = [ids[0], ids[0]]
    if config.join_algorithm == "naive_uniform":
        return run_uniform(oracle, config, ids)
    elif config.join_algorithm == "naive_importance":
        # scores = _preprocess_scores(scores)
        return run_naive_importance(scores[0], oracle, config, ids)
    elif config.join_algorithm == "weighted_wander":
        return run_weighted_wander(scores, oracle, config, ids)
    elif config.join_algorithm == "ripple":
        return run_ripple(oracle, config, ids)
    else:
        raise NotImplementedError(f"join method {config.join_algorithm} not implemented")
    
if __name__ == "__main__":
    # test preprocess scores
    from joinml.utils import normalize
    import pickle
    from joinml.dataset_loader import JoinDataset
    config = Config(
        data_path="../../data",
        dataset_name="quora",
        join_algorithm="naive_importance",
        proxy="Cosine",
        is_self_join=True,
        log_path="logs/quora-ni-cosine.log",
        repeats=20,
        proxy_cache=True
    )
    dataset = JoinDataset(config)
    ids = dataset.get_ids()[0]
    # print(ids)
    with open("../../data/quora/proxy_cache/Cosine.pkl", "rb") as f:
        scores = pickle.load(f)
    for i in range(len(scores)):
        scores[i] = normalize(scores[i], is_self_join=True)
    scores = _preprocess_scores(scores)
    print(scores.shape)
    max_score = np.max(scores)
    print(max_score)
    candidate = np.array(np.where(scores >= 0.01 * max_score)).T
    candidate_ids = [[ids[c[0]], ids[c[1]]] for c in candidate ]
    print(len(candidate_ids))
    oracle = Oracle(config)
    count = 0
    for candidate_id in candidate_ids:
        if oracle.query(candidate_id):
            count += 1
    print(count)
