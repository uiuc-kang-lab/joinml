import argparse
import logging
from naive_random import run_random
from dataset_loader import load_dataset
from ripple_join import run_ripple
import time
from importance_sampling import run_importance
from weighted_wander import run_weighted_wander

def run(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", 
                        datefmt="%Y-%m-%d %H:%M:%S", filename=f"logs/run_{args.dataset}_{args.method}_{time.time()}.log")
    logging.info(f"args: {args}")
    sample_ratios = [0.00001*x for x in range(1,100)] + [0.001*x for x in range(1,100)] + [0.2, 0.5, 0.75, 1]

    dataset = load_dataset(args.dataset, args.proxy, args.limit)

    if args.method == "naive_random":
        run_random(sample_ratios, dataset, args.repeats, f"results/{args.dataset}-{args.method}.csv", 
                   seed=args.seed, num_worker=args.num_worker)
    elif args.method == "ripple_join":
        run_ripple(sample_ratios, dataset, args.repeats, f"results/{args.dataset}-{args.method}.csv", 
                   seed=args.seed, num_worker=args.num_worker)
    elif args.method == "naive_importance":
        run_importance(sample_ratios, dataset, args.repeats,
                       f"results/{args.dataset}-{args.method}.csv", seed=args.seed, num_worker=args.num_worker)
    elif args.method == "weighted_wander_join":
        run_weighted_wander(sample_ratios, dataset, args.repeats,
                            f"results/{args.dataset}-{args.method}.csv", seed=args.seed, num_worker=args.num_worker)
    else:
        raise NotImplementedError(f"method {args.method} is not implemented")


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

