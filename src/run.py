import argparse
from naive_random import run_random
from dataset_loader import load_dataset
from ripple_join import run_ripple
from importance_sampling import run_importance
from weighted_wander import run_weighted_wander

def run(args: argparse.Namespace):
    sample_ratios = [0.00001*x for x in range(1,100)] + [0.001*x for x in range(1,100)] + [0.2, 0.5, 0.75, 1]

    ltable, rtable, labels, proxy = load_dataset(args.dataset, args.proxy, args.limit)

    if args.method == "naive_random":
        run_random(sample_ratios, ltable, rtable, labels, args.repeats, 
                   f"results/{args.dataset}-{args.method}.csv", seed=args.seed, num_worker=args.num_worker)
    elif args.method == "ripple_join":
        run_ripple(sample_ratios, ltable, rtable, labels, args.repeats,
                   f"results/{args.dataset}-{args.method}.csv", seed=args.seed, num_worker=args.num_worker)
    elif args.method == "naive_importance":
        run_importance(sample_ratios, ltable, rtable, labels, proxy, args.repeats,
                       f"results/{args.dataset}-{args.methods}.csv", seed=args.seed, num_worker=args.num_worker)
    elif args.method == "weighted_wander_join":
        run_weighted_wander(sample_ratios, ltable, rtable, labels, proxy, args.repeats,
                            f"results/{args.dataset}-{args.methods}.csv", seed=args.seed, num_worker=args.num_worker)
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

