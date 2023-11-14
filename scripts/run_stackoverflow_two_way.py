from joinml.config import Config
from joinml.executable.run_two_way_sampling import run
config = Config(
    dataset_name="stackoverflow",
    device="cpu",
    is_self_join=True,
    dataset_cutoff=[1000000, 2000000],
    upper_sample_size=[500000],
    lower_sample_size=[1000000, 2000000, 3000000, 4000000, 5000000, 6000000, 700000, 8000000],
    repeats=50,
    log_path="logs/stackoverflow-two_way-LM.log",
    cached_blocking=True,
    proxy="data/stackoverflow_proxy",
    large_data_path=".",
    enable_remap=False
)

run(config)
