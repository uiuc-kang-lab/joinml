import argparse
import logging
from joinml.config import Config
from joinml.algs.recall import run as run_recall
from joinml.algs.standard_sampling import run as sampling
from joinml.algs.blocking_noci import run as blocking_noci
from joinml.algs.joinml_fixed import run as joinml_fixed
from joinml.algs.joinml_adapt import run as joinml_adapt
from joinml.utils import set_random_seed

def run(config: Config):
    print(f"running job {config.seed}")
    set_random_seed(config.seed)
    if config.task == "recall":
        run_recall(config)
    elif config.task in ["importance", "uniform"]:
        sampling(config)
    elif config.task == "blocking-noci":
        blocking_noci(config)
    elif config.task == "joinml-fixed":
        joinml_fixed(config)
    elif config.task == "joinml-adapt":
        joinml_adapt(config)
    else:
        raise ValueError(f"Unknown task: {config.task}")
