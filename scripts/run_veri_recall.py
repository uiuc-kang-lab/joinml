from joinml.run import run
from joinml.config import Config

config = Config(
    data_path="./data",
    cache_path="../.cache/joinml",
    log_path="logs/veri_recall.log",
    dataset_name="VeRi",
    proxy="reid",
    is_self_join=False,
    proxy_score_cache=True,
    task="recall",
    device="cuda",
    batch_size=16
)

run(config) 