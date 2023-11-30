from dataclasses import dataclass

@dataclass
class Config:
    """Configuration class for JoinML."""
    data_path: str = "data"
    cache_path: str = "."
    model_cache_path: str = f"{cache_path}/model_cache"
    dataset_name: str = "twitter"
    seed: int = 233
    proxy: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    log_path: str = f"logs/test.log"
    batch_size: int = 4
    is_self_join: bool = True
    repeats: int=20
    confidence_level: float=0.95

    # cache knobs
    proxy_score_cache: bool = True
    blocking_cache: bool = True

    # proxy evaluation
    proxy_eval_data_sample_size: float = 10000
    proxy_eval_data_sample_method: str = "random"

    # algorithm related
    blocking_size: int = 1000000
    sample_size: int = 1000000
    defensive_rate: float = 0

    # special
    enable_remap: bool = True
    proxy_normalizing_style: str = "proportional"

    oracle_budget: int = -1
