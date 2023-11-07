from joinml.config import Config
from joinml.proxy.get_proxy import get_proxy_by_evaluation
from joinml.utils import set_random_seed, set_up_logging, normalize
from joinml.dataset_loader import JoinDataset
from joinml.oracle import Oracle

def run(config: Config):
    # set up
    set_up_logging(config.log_path)
    set_random_seed(config.seed)
    # dataset
    dataset = JoinDataset(config)
    # oracle
    oracle = Oracle(config)
    get_proxy_by_evaluation(config, dataset, oracle)
    
