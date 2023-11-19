from joinml.config import Config
from joinml.executable.run_uniform import run

config = Config(
    cache_path="/mydata/yuxuan",
    dataset_name="company",
    is_self_join=False,
    log_path="logs/company-uniform.log",
    repeats=50,
    sample_size=[10000000, 5000000, 1000000, 500000, 100000]
)

run(config)