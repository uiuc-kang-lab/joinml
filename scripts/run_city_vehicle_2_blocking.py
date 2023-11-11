from joinml.executable.run_blocking import run
from joinml.config import Config
import logging

config = Config(
    dataset_name="city_vehicle_2",
    join_algorithm="naive_importance",
    proxy="pHash",
    is_self_join=False,
    log_path="logs/city_vehicle-blocking_pHash.log",
    repeats=20,
    proxy_cache=True,
    device="cpu"
)

run(config)
