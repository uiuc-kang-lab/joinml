from joinml.config import Config
from joinml.executable.run_uniform import run

config = Config(
    cache_path="/mydata/yuxuan",
    dataset_name="stackoverflow",
    is_self_join=True,
    log_path="logs/stackoverflow-uniform.log",
    repeats=50,
    sample_size=[1000000, 2000000, 3000000, 4000000, 5000000]
)

run(config)