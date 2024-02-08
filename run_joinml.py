from joinml.run import run
from joinml.config import Config
import argparse
import yaml
import time

def parse_args() -> Config:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_config", type=str, default="configs/company.yaml")
    parser.add_argument("--task", type=str, default="joinml-adapt")
    parser.add_argument("--data_path", type=str, default="data")
    parser.add_argument("--cache_path", type=str, default="../.cache/joinml")
    parser.add_argument("--oracle_budget", type=int, default=1000000)
    parser.add_argument("--max_blocking_ratio", type=float, default=0.2)
    parser.add_argument("--internal_loop", type=int, default=500)
    parser.add_argument("--log_level", type=str, default="info")
    parser.add_argument("--blocking_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=int(time.time()))
    args = parser.parse_args()
    with open(args.dataset_config, "r") as f:
        config = yaml.safe_load(f)
    for k, v in vars(args).items():
        if k != "dataset_config":
            config[k] = v
    config["output_file"] = f"{config['dataset_name']}-{config['task']}.jsonl"
    config["log_path"] = f"logs/{config['dataset_name']}-{config['task']}_{config['seed']}.log"
    return Config(**config)

if __name__ == "__main__":
    config = parse_args()
    run(config)