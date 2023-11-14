from dataclasses import dataclass, field
import time

@dataclass
class Config:
    """Configuration class for JoinML."""
    data_path: str = "data"
    model_cache_path: str = "model_cache"
    dataset_name: str = "twitter"
    seed: int = 233
    proxy: str = "all-MiniLM-L6-v2"
    device: str = "mps"
    log_path: str = f"logs/test.log"
    batch_size: int = 4
    is_self_join: bool = True
    repeats: int=20
    join_algorithm: str = "naive_uniform"
    join_sample_size: int=5000
    confidence_level: float=0.95
    proxy_cache: bool = True
    
    # proxy evaluation
    proxy_eval_data_sample_size: float = 10000
    proxy_eval_data_sample_method: str = "random"

    # proxy finetuning arguments
    train_data_sample_size: float = 1000
    train_data_sample_method: str = "random"
    embedding_dim: int = 128
    head: str = "mlp"
    temp: float = 0.07
    epoch: int = 10

    # proxy blocking
    lower_sample_size: int = 100000
    upper_sample_size: int = 100000 
    dataset_cutoff: int = 100000
    cached_blocking: bool = False
    
    # path
    large_data_path: str = "."

    # special
    enable_remap: bool = True