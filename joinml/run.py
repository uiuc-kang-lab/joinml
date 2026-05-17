import logging

from joinml.config import Config
from joinml.algs.recall import run as run_recall
from joinml.algs.standard_sampling import run as sampling
from joinml.algs.blocking_noci import run as blocking_noci
from joinml.algs.blocking_ci import run as blocking_ci
from joinml.algs.joinml_fixed import run as joinml_fixed
from joinml.algs.joinml_adapt import run as joinml_adapt
from joinml.algs.wanderjoin import run as wanderjoin
from joinml.algs.joinml_select import run as joinml_recall
from joinml.algs.standard_sampling_select import run as sampling_select
from joinml.algs.wanderjoin_select import run as wanderjoin_select
from joinml.algs.joinml_scale import run as joinml_scale
from joinml.algs.sampling_scale import run as sampling_scale
from joinml.algs.blocking_scale import run as blocking_scale
from joinml.algs.joinml_topk import run as joinml_topk
from joinml.algs.sampling_topk import run as sampling_topk
from joinml.utils import set_random_seed


def run(config: Config):
    """Dispatch a single experiment by `config.task` name.

    Supported tasks:
      - joinml-adapt         : Blocking-augmented Sampling (BaS, the default)
      - joinml-fixed         : BaS with a fixed (non-adaptive) blocking ratio
      - importance / uniform : standard importance / uniform sampling baselines
      - wanderjoin           : Weighted Wander Join baseline
      - blocking-noci        : embedding-based blocking, no CI
      - blocking-ci          : blocking with confidence intervals
      - recall               : recall@k oracle traversal
      - joinml-recall / -precision, importance-select, uniform-select,
        wanderjoin-select    : selection-query variants (target recall/precision)
      - joinml-topk, uniform-topk, importance-topk : Top-K heavy hitters
      - large-join, uniform-scale, importance-scale, blocking-noci-scale
                             : scaled variants for very large cross products
    """
    logging.info("Starting task=%s seed=%s", config.task, config.seed)
    set_random_seed(config.seed)
    if config.task == "recall":
        run_recall(config)
    elif config.task in ("importance", "uniform"):
        sampling(config)
    elif config.task == "blocking-noci":
        blocking_noci(config)
    elif config.task == "blocking-noci-scale":
        blocking_scale(config)
    elif config.task == "blocking-ci":
        blocking_ci(config)
    elif config.task == "joinml-fixed":
        joinml_fixed(config)
    elif config.task == "joinml-adapt":
        joinml_adapt(config)
    elif config.task == "wanderjoin":
        wanderjoin(config)
    elif config.task == "wanderjoin-select":
        wanderjoin_select(config)
    elif config.task == "joinml-recall":
        joinml_recall(config)
    elif config.task in ("joinml-precision", "importance-select", "uniform-select"):
        sampling_select(config)
    elif config.task == "large-join":
        joinml_scale(config)
    elif config.task in ("uniform-scale", "importance-scale"):
        sampling_scale(config)
    elif config.task == "joinml-topk":
        joinml_topk(config)
    elif config.task in ("uniform-topk", "importance-topk"):
        sampling_topk(config)
    else:
        raise ValueError(f"Unknown task: {config.task}")
