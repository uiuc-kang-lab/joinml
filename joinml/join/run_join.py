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
        print(expanded_axies)
        if i == 0:
            all_scores = np.expand_dims(scores[i], axis=tuple(expanded_axies))
        else:
            all_scores = all_scores * np.expand_dims(scores[i], axis=tuple(expanded_axies))
        print(np.sum(all_scores))
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
        scores = _preprocess_scores(scores)
        return run_naive_importance(scores, oracle, config, ids)
    elif config.join_algorithm == "weighted_wander":
        return run_weighted_wander(scores, oracle, config, ids)
    elif config.join_algorithm == "ripple":
        return run_ripple(scores, oracle, config, ids)
    else:
        raise NotImplementedError(f"join method {config.join_algorithm} not implemented")
    
if __name__ == "__main__":
    # test preprocess scores
    from joinml.utils import normalize
    scores = [np.random.rand(100, 200), np.random.rand(200, 300)]
    for i in range(len(scores)):
        scores[i] = normalize(scores[i])
        print(np.sum(scores[i]))
    scores = _preprocess_scores(scores)
    print(np.sum(scores))
