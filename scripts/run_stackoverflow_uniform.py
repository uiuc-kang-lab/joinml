from joinml.config import Config
from joinml.executable.run_uniform import run

config = Config(
    cache_path="/mydata/yuxuan",
    dataset_name="stackoverflow",
    is_self_join=True,
    log_path="logs/stackoverflow-uniform.log",
    repeats=50,
    sample_size=[100000000, 50000000, 10000000, 5000000, 1000000, 500000, 100000]
)

run(config)