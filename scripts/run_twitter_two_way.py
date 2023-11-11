from joinml.config import Config
from joinml.executable.run_two_way_sampling import run

config = Config(
    dataset_name="twitter",
    device="cpu",
    is_self_join=True,
    dataset_cutoff=[10000, 50000, 100000, 500000, 1000000],
    upper_sample_size=[100000, 200000, 300000, 400000, 500000],
    lower_sample_size=[200000*i for i in range(1, 6)],
    repeats=20,
    log_path="logs/twitter-two_way-LM.log",
)

run(config)
