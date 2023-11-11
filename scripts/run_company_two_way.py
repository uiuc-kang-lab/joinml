from joinml.config import Config
from joinml.executable.run_two_way_sampling import run
config = Config(
    dataset_name="company",
    device="cpu",
    is_self_join=False,
    dataset_cutoff=[1000, 5000, 10000, 50000, 100000, 500000, 1000000],
    upper_sample_size=[10000, 50000, 100000, 500000],
    lower_sample_size=[100000, 500000, 1000000, 5000000],
    repeats=20,
    log_path="logs/company-two_way-LM.log",
)

run(config)