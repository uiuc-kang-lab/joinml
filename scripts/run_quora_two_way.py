from joinml.config import Config
from joinml.executable.run_two_way_sampling import run
config = Config(
    dataset_name="quora",
    device="cpu",
    is_self_join=True,
    dataset_cutoff=[100000, 500000, 1000000],
    upper_sample_size=[10000],
    lower_sample_size=[1000000, 500000, 100000],
    repeats=20,
    log_path="logs/quora-two_way-LM.log",
    cached_blocking=False
)

run(config)
