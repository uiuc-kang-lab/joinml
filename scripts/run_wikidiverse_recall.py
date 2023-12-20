from joinml.run import run
from joinml.config import Config

config = Config(
    data_path="/home/yxx404/scratch",
    cache_path="../.cache/joinml",
    log_path="logs/wikidiverse_recall.log",
    dataset_name="wikidiverse",
    proxy="ViT-B/32",
    is_self_join=False,
    proxy_score_cache=True,
    task="recall",
    device="cpu",
    batch_size=16
)

run(config)