from joinml.run import run
from joinml.config import Config

config = Config(
    data_path="./data",
    cache_path="../.cache/joinml",
    log_path="logs/flickr30k_recall.log",
    dataset_name="flickr30k",
    proxy="blip",
    is_self_join=False,
    proxy_score_cache=True,
    task="recall",
    device="cpu",
    batch_size=16
)

run(config)