import argparse
import logging
from joinml.config import Config
from joinml.algs.recall import run as run_recall
from joinml.algs.standard_sampling import run as sampling
from joinml.algs.blocking_noci import run as blocking_noci
from joinml.algs.joinml_fixed import run as joinml_fixed
from joinml.algs.joinml_adapt import run as joinml_adapt
from joinml.algs.wanderjoin import run as wanderjoin
from joinml.algs.post_hoc_var import run as post_hoc
from joinml.algs.joinml_select import run as joinml_recall
from joinml.algs.standard_sampling_select import run as sampling_select
from joinml.algs.wanderjoin_select import run as wanderjoin_select
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
    elif config.task == "BaS":
        joinml_adapt(config)
    elif config.task == "wanderjoin":
        wanderjoin(config)
    elif config.task == "post-hoc":
        post_hoc(config)
    elif config.task == "wanderjoin-select":
        wanderjoin_select(config)
    elif config.task == "joinml-recall":
        joinml_recall(config)
    elif config.task in ["joinml-precision", "importance-select", "uniform-select"]:
        sampling_select(config)
    else:
        raise ValueError(f"Unknown task: {config.task}")

