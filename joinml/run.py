import argparse
import logging
from joinml.config import Config
from joinml.executable.recall import run as run_recall
from joinml.executable.straight_sampling import run as run_straight_sampling
from joinml.executable.blocking_sampling import run as run_blocking_sampling

def run(config: Config):
    if config.task == "recall":
        run_recall(config)
    if config.task in ["is", "uniform"]:
        run_straight_sampling(config)
    if config.task == "bis":
        run_blocking_sampling(config)
    else:
        raise ValueError(f"Unknown task: {config.task}")
