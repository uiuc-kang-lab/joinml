# JoinML

## Setup

Step 1: Install python and packages
```bash
$ python --version
Python 3.10.13
$ pip install -r requirements.txt
$ pip install -e .
```

Step 2: Download datasets from google drive
```bash
$ cd data
$ bash download.sh download <dataset_name>
```

Step 3: Setup environment variable for proxy cache (in anaconda)
```bash
$ conda env config vars set <my_var> = "path_to_cache"
$ conda activate <env_name>
```

## Run experiments

Step 1: set up config, see examples in the `./scripts`

Step 2: run the config script in the proper environment
```bash
$ python scripts/<script_name>.py
```

## Datasets

Check `./data/download.sh` for downloading and removing datasets.

| Dataset       | task                               | table size   | join size | positive rate | reference |
|---------------|------------------------------------|--------------|-----------|---------------|-----------|
| twitter       | duplicate detection                | 100000       | 10G       | 1.77e-7       | [[1]](https://languagenet.github.io/) |
| quora         | duplicate detection                | 60000        | 3600M     | 1.02e-6       | [[1]](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs) [[2]](https://aclanthology.org/2020.acl-main.197/) |
| company       | entity matching                    | 10000,10000  | 100M      | 3.55e-5       | [[1]](https://dl.acm.org/doi/10.1145/3183713.3196926) [[2]](https://arxiv.org/abs/2004.00584) |
| stackoverflow | duplicate detection                | 60000        | 3600M     | 1.70e-6       | [[1]](http://2013.msrconf.org/challenge.php#challenge_data) [[2]](https://link.springer.com/article/10.1007/s11390-015-1576-4) |
| city_human    | multi-camera multi-target tracking | 23542,14653  | 345M      | 0.14          | [[1]](https://www.aicitychallenge.org/)|
| flickr30K     | multi-modal entity matching        | 31783,158915 | 5G        | 3.1e-5        | [[1]](https://github.com/BryanPlummer/flickr30k_entities)|


