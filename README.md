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
$ conda env config vars set cahce_path = "path_to_cache"
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

| Dataset       | task                               | table size   | join size | matching rate | reference |
|---------------|------------------------------------|--------------|-----------|---------------|-----------|
| company       | entity matching                    | 10000,10000  | 100M      | 3.6e-5        | [[1]](https://dl.acm.org/doi/10.1145/3183713.3196926) [[2]](https://arxiv.org/abs/2004.00584) |
| quora         | duplicate detection                | 60000        | 3G        | 1.1e-6        | [[1]](https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs) [[2]](https://aclanthology.org/2020.acl-main.197/) |
| webmasters    | duplicate detection                | 142704       | 20G       | 4.5e-7        | [[1]](http://2013.msrconf.org/challenge.php#challenge_data) [[2]](https://link.springer.com/article/10.1007/s11390-015-1576-4) |
| veri          | multi-camera multi-target tracking | 21150,28175  | 595M      | 1.4e−3        | [[1]](https://link.springer.com/chapter/10.1007/978-3-319-46475-6_53) [[2]](https://ieeexplore.ieee.org/abstract/document/8036238?casa_token=EOusFmxPGPEAAAAA:UivZ3_pZADRbwEoLMGzZ2HWC_Sdlbw7T-S4AMQr5zChy-VebeBsXdEmsdRUxT6a9ENxB1KGI) |
| city_human    | multi-camera multi-target tracking | 23542,14653  | 345M      | 0.15          | [[1]](https://www.aicitychallenge.org/)|
| flickr30K     | multi-modal entity matching        | 31783,158915 | 5G        | 3.1e-5        | [[1]](https://github.com/BryanPlummer/flickr30k_entities)|


