from dataclasses import dataclass

@dataclass
class Config:
    """Configuration class for JoinML."""
    data_path: str = "data"
    cache_path: str|None = "."
    model_cache_path: str = f"{cache_path}/models"
    log_path: str = "logs/test.log"
    dataset_name: str = "twitter"
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
    num_strata: int = 11
    bootstrap_trials: int = 10000 # [0, 10000] bis/(is/uniform avg) 10000, for others both works
    task: str = "bis" # [bis, is, uniform, recall]
    output_file: str = f"{dataset_name}.jsonl"
    log_level: str = "info" # [info, debug]
    parallelProxyCalculation: bool = True
    numProcessPerCPU = 1 #can be non integer
