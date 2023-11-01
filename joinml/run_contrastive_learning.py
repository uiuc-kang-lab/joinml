from joinml.config import Config
from joinml.oracle import Oracle
from joinml.proxy.embedding_proxy import TransformerProxy
from joinml.sampling.training_data_sample import get_train_data
from joinml.dataset_loader import JoinDataset
from joinml.proxy.evaluation import Evaluator
from joinml.proxy.finetune_embedding import Trainer
from joinml.utils import set_random_seed

import logging
from torch.utils.data import DataLoader


def run(config: Config):
    set_random_seed(config.seed)

    # prepare logging
    logging.basicConfig(filename=config.log_path, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        filemode='w')

    # prepare oracle
    oracle = Oracle(config)

    # prepare proxy
    proxy = TransformerProxy(config)
    
    dataset = JoinDataset(config)

    # prepare training data, label
    train_dataset = get_train_data(config, dataset, oracle, tokenizer=proxy.get_tokenizer())
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    
    # prepare the evaluator
    # logging.info("prepare the evaluator")
    # evaluator = Evaluator(config, dataset, oracle)

    # evaluate pre-trained model
    # logging.info(f"evaluate pre-trained model: {config.proxy}")
    # improvement = evaluator.evaluate(proxy)
    # logging.info(f"improvement: {improvement}")

    # train model
    logging.info(f"fine-tune model: {config.proxy}")
    trainer = Trainer(proxy.model, proxy.model.get_sentence_embedding_dimension(), config.embedding_dim, head=config.head, temp=config.temp)
    model, loss = trainer.train(train_loader=train_loader, epochs=config.epoch, device=config.device)
    logging.info("loss: {}".format(loss))
    proxy.model = model

    # evaluate model
    logging.info(f"evaluate fine-tuned model: {config.proxy}")
    # improvement = evaluator.evaluate(proxy)
    # logging.info(f"improvement: {improvement}")