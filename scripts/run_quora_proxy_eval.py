from joinml.config import Config
from joinml.executable.run_proxy_eval import run

config = Config(
    dataset_name="quora",
    is_self_join=True,
    log_path="logs/quora-proxy_eval.log",
    device="cpu",
    proxy_eval_data_sample_size=10000,
    proxy_eval_data_sample_method="random"
)

run(config)