from dataclasses import dataclass
import time

@dataclass
class Config:
    """Configuration class for JoinML."""
    data_path: str = "data"
    dataset_name: str = "twitter"
    train_data_sample_rate: float = 1e-5
    train_data_sample_method: str = "random"
    proxy_eval_data_sample_rate: float = 1e-5
    proxy_eval_data_sample_method: str = "random"
    seed: int = 233
    proxy: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    log_path: str = f"logs/test.log"
    batch_size: int = 4
    is_self_join: bool = True
    embedding_dim: int = 128
    head: str = "mlp"
    temp: float = 0.07
    epoch: int = 10
