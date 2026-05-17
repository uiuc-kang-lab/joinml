from dataclasses import dataclass
from typing import Union
@dataclass
class Config:
    """Configuration class for JoinML."""
    data_path: str = "data"
    cache_path: Union[str,None] = "."
    model_cache_path: str = f"{cache_path}/models"
    log_path: str = "logs/test.log"
    dataset_name: str = "company"
    seed: int = 233
    proxy: str = "TF/IDF"
    device: str = "cpu"
    batch_size: int = 4
    is_self_join: bool = True
    confidence_level: float=0.95
    proxy_score_cache: bool = True
    proxy_normalizing_style: str = "proportional" # [sqrt, proportional]
    oracle_budget: int = 1000000
    max_blocking_ratio: float = 0.2
    bootstrap_trials: int = 1000 # [1000, 10000]
    task: str = "joinml-adapt" # [joinml-adapt, joinml-fix, is, uniform, recall]
    output_file: str = f"{dataset_name}.jsonl"
    log_level: str = "info" # [info, debug]
    parallelProxyCalculation: bool = True
    numProcessPerCPU = 1 #can be non integer
    blocking_ratio: float = 0.2 # only used for fix blocking sampling
    internal_loop: int = 1 # for time efficiency of large datasets
    aggregator: str = "count" # [count, sum, avg]
    target: float = 0.95 # target recall for recall/precision task
    ci: bool = False # confidence interval for recall/precision task
    w_exp: float = 1.0 # weight exponent
    table_ids: str = "0,1"
    join_reorder: bool = False
    top_k: int = 5 # for top-k task
    # Phase-C ablation knobs; defaults preserve Phase-B behaviour.
    defensive_mix_ratio: float = 0.0 # (1-ε)W + ε·uniform; 0 disables
    allocation_search: str = "prefix" # "prefix" | "subset" | "auto" | "evt"
    sampling_scheme: str = "wr" # "wr" (with replacement) | "wor"
    strata_size: int = 1000 # target pilot-sample size per blocking-candidate stratum
    var_shrinkage: float = 0.0 # Q1: pseudo-samples worth of pooled-variance prior; 0 disables
    force_block_concentrated: bool = False # Q2: force β={K} when stratum-K match-rate >> stratum-0
    force_block_threshold: float = 100.0 # match-rate ratio threshold for Q2
    two_stage_allocation: bool = False # Q5: pilot → rough β → 30% main → re-allocate → 70% main