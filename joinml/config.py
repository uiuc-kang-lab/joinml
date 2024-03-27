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