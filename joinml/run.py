import argparse
import logging
from joinml.config import Config
from joinml.executable.recall import run as run_recall
from joinml.executable.straight_sampling import run as run_straight_sampling
from joinml.executable.joinml_dep2 import run as run_blocking_sampling
from joinml.executable.straight_blocking import run as run_straight_blocking
from joinml.executable.fix_blocking_sampling import run as run_fix_blocking_sampling
from joinml.executable.joinml_dep1 import run as run_joinml
from joinml.executable.joinml import run as run_joinml_stats_guarantees
from joinml.utils import set_random_seed

def run(config: Config):
    print(f"running job {config.seed}")
    set_random_seed(config.seed)
    if config.task == "recall":
        run_recall(config)
    elif config.task in ["is", "uniform"]:
        run_straight_sampling(config)
    elif config.task == "blocking":
        run_straight_blocking(config)
    elif config.task == "fb":
        run_fix_blocking_sampling(config)
    elif config.task == "joinml":
        run_joinml_stats_guarantees(config)
    else:
        raise ValueError(f"Unknown task: {config.task}")
