from joinml.config import Config
from joinml.executable.run_two_way_sampling import run
config = Config(
    dataset_name="company",
    device="cpu",
    is_self_join=False,
    dataset_cutoff=[1000000],
    upper_sample_size=[500000],
    lower_sample_size=[1000000],
    repeats=20,
    log_path="logs/company-two_way-LM.log",
)

run(config)