import argparse
import logging
from joinml.config import Config
from joinml.executable.recall import run as run_recall

def run(config: Config):
    if config.task == "recall":
        run_recall(config)
    else:
        raise NotImplementedError
