from joinml.executable.run_uniform import run
from joinml.config import Config

config = Config(
    dataset_name="city_vehicle_2",
    is_self_join=False,
    log_path="logs/city_vehicle-uniform.log",
    repeats=50,
    device="cpu",
    cache_path="/mydata/yuxuan",
    sample_size=[5000000, 2500000, 1000000, 750000, 500000, 250000, 100000],
)

run(config)
