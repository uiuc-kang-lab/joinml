from joinml.config import Config
from joinml.executable.run_random import run
config = Config(
    dataset_name="stackoverflow",
    device="cpu",
    is_self_join=True,
    lower_sample_size=[1000000*i for i in range(1, 11)],
    repeats=50,
    log_path="logs/stackoverflow-random.log",
)

run(config)
