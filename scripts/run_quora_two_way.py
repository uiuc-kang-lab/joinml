from joinml.config import Config
from joinml.executable.run_two_way_sampling import run
config = Config(
    dataset_name="quora",
    device="cpu",
    is_self_join=True,
    dataset_cutoff=[500, 1000, 5000, 8000],
    upper_sample_size=[100, 500, 1000, 5000, 8000],
    lower_sample_size=[10000, 100000, 50000, 100000, 500000, 1000000, 5000000],
    repeats=20,
    log_path="logs/quora-two_way-LM.log",
)

run(config)
