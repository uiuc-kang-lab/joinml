from dataclasses import dataclass

@dataclass
class Config:
    """Configuration class for JoinML."""
    data_path: str = "data"
    cache_path: str = "."
    model_cache_path: str = f"{cache_path}/models"
    log_path: str = "logs/test.log"
    dataset_name: str = "twitter"
    seed: int = 233
    proxy: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 4
    is_self_join: bool = True
    confidence_level: float=0.95
    proxy_score_cache: bool = True
    proxy_normalizing_style: str = "proportional"
    oracle_budget: int = 1000000
    num_strata: int = 11
    bootstrap_trials: int = 10000
    task: int = "bis" # [bis, is, uniform, recall]
