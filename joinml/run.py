import argparse
import logging
from joinml.config import Config
from joinml.executable.recall import run as run_recall

def run(config: Config):
    if config.task == "recall":
        run_recall(config)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, default="naive_random")
    parser.add_argument("--dataset", type=str, default="qqp")
    parser.add_argument("--proxy", type=str, default="cosine")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--num_worker", type=int, default=1)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=2333)
    args = parser.parse_args()
    run(args)

