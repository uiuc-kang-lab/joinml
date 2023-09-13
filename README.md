# Datasets

Real data are stored on Google drive https://drive.google.com/drive/folders/19MnzAFLazqqkTQNXOPp7mGwjLE4MEIVr?usp=drive_link

## Company

**Description**: This dataset is proposed at [SIGMOD'18](https://dl.acm.org/doi/10.1145/3183713.3196926) and recently studied by [Ditto](https://arxiv.org/abs/2004.00584) with LLMs.

**Task**: find the all the matching entities between two datasets.

**Dataset size**:

| Table A | Table B | pairs |
|-|-|-|
|28.2K|28.2K|800M|

**Oracle model**: RoBerTa (check [Ditto's code](https://github.com/megagonlabs/ditto) for training)

**Proxy**: 
1. [string similarity measure](https://sites.google.com/site/anhaidgroup/current-projects/magellan/py_stringmatching?authuser=0)
2. Logistic regression model, [training scripts](https://github.com/cocoxu/SemEval-PIT2015) implemented by [SemEval](https://aclanthology.org/S15-2001/). 
3. BiMPM (no pretrained released, need to train by ourselfes, scripts released at the [github repo](https://github.com/zhiguowang/BiMPM) )
4. Siamese-CNN (no pretrained, train script released, need to construct architeccture and train)
5. Pair-CNN [paper](https://dl-acm-org.proxy2.library.illinois.edu/doi/abs/10.1145/2766462.2767738)

### TODO:
- [ ] Clean the company data to two csv files. Each csv file has three columns (id, url, content), where id is a unique integer identifier. Upload the clean data to google drive (folder preprocessed/).
- [ ] (report estimated completion time) finetuning RoBerTa / find find-tuned RoBerTa according to Ditto's codebase https://github.com/megagonlabs/ditto 
- [ ] (report estimated completion time) run fine-tuned RoBerTa on the whole dataset and calculate the accuracy
- [ ] (report estimated completion time) run string similarity measure algorithm on the company data and calculate accuracy

## Quora Question Pair (QQP)

**Description**: This dataset is released by Quora.com and recently studied by [SMART](https://aclanthology.org/2020.acl-main.197/).

**Task**: find all the duplicate questions.

**Dataset size**:
| Table | pairs |
|-|-|
|8M|8MM|

**Oracle model**: 
1. [fine-tuned roberta](https://huggingface.co/AMHR/adversarial-paraphrasing-detector)
2. SMART_BERT (claimed to be released at [github repo](https://github.com/namisan/mt-dnn))

**Proxy**:
1. [string similarity measure](https://sites.google.com/site/anhaidgroup/current-projects/magellan/py_stringmatching?authuser=0)
2. Logistic regression model, [training scripts](https://github.com/cocoxu/SemEval-PIT2015) implemented by [SemEval](https://aclanthology.org/S15-2001/). 
3. BiMPM (no pretrained released, need to train by ourselfes, scripts released at the [github repo](https://github.com/zhiguowang/BiMPM) )
4. Siamese-CNN (no pretrained, train script released, need to construct architeccture and train)
5. Pair-CNN [paper](https://dl-acm-org.proxy2.library.illinois.edu/doi/abs/10.1145/2766462.2767738)

### TODO (Yuxuan):
- [x] Clean the Quora data to two csv files. Each csv file has three columns (id, url, content), where id is a unique integer identifier. Upload the clean data to google drive (folder preprocessed/).
- [x] (report estimated completion time) finetuning RoBerTa / find find-tuned RoBerTa
- [ ] (report estimated completion time) run fine-tuned RoBerTa on the whole dataset and calculate the accuracy
- [ ] (report estimated completion time) run string similarity measure algorithm on the company data and calculate accuracy

## Stack Overflow

**Description**: This dataset is an offocial dump released by [MSR2013](http://2013.msrconf.org/challenge.php#challenge_data), where the duplication issue is [studied](https://link.springer.com/article/10.1007/s11390-015-1576-4).

**Task**: label all the duplicate posts.

**Dataset size**:
| Table | pairs |
|-|-|
| 10M | 100MM |

**Oracle model**: fine-tuned LLM (https://github.com/beir-cellar/beir)

**Proxy**:
1. [string similarity measure](https://sites.google.com/site/anhaidgroup/current-projects/magellan/py_stringmatching?authuser=0)
2. Logistic regression model, [training scripts](https://github.com/cocoxu/SemEval-PIT2015) implemented by [SemEval](https://aclanthology.org/S15-2001/). 
3. BiMPM (no pretrained released, need to train by ourselfes, scripts released at the [github repo](https://github.com/zhiguowang/BiMPM) )
4. Siamese-CNN (no pretrained, train script released, need to construct architeccture and train)
5. Pair-CNN [paper](https://dl-acm-org.proxy2.library.illinois.edu/doi/abs/10.1145/2766462.2767738)

### TODO:
- [ ] Clean the StackOverflow data to one csv file with three columns (id, title, body), where id is a unique integer identifier. Upload the clean data to google drive (folder preprocessed/).
- [ ] (report estimated completion time) finetuning / find LLM from BEir
- [ ] (report estimated completion time) run fine-tuned LLM on the whole dataset and calculate the accuracy
- [ ] (report estimated completion time) run string similarity measure algorithm on the data and calculate accuracy

## LanguageNet

**Description**: [LanguageNet](https://languagenet.github.io/) is a collection of sentence level paraphrases from Twitter by linking tweets through shared URLs.

**Task**: find all paraphrased sentences

**Dataset size**:
| Table 1 | Table 2 | pairs |
|-|-|-|
|51K|51K|2B|

**Oracle model**: fine-tune Roberta from https://huggingface.co/AMHR/adversarial-paraphrasing-detector

**Proxy**:
1. [string similarity measure](https://sites.google.com/site/anhaidgroup/current-projects/magellan/py_stringmatching?authuser=0)
2. Logistic regression model, [training scripts](https://github.com/cocoxu/SemEval-PIT2015) implemented by [SemEval](https://aclanthology.org/S15-2001/). 
3. BiMPM (no pretrained released, need to train by ourselfes, scripts released at the [github repo](https://github.com/zhiguowang/BiMPM) )
4. Siamese-CNN (no pretrained, train script released, need to construct architeccture and train)
5. Pair-CNN [paper](https://dl-acm-org.proxy2.library.illinois.edu/doi/abs/10.1145/2766462.2767738)

### TODO:
- [ ] Clean the LanguageNet data to one csv file with two columns (id, content), where id is a unique integer identifier. Upload the clean data to google drive (folder preprocessed/).
- [ ] (report estimated completion time) run string similarity measure algorithm on the data and calculate accuracy
- [ ] (report estimated completion time) run fine-tuned RoBerTa on the whole dataset and calculate the accuracy

## Carla
working on it

## ArXiv

Description: to be added

Dataset size: to be added

Oracle model: fine-tune Roberta from https://huggingface.co/AMHR/adversarial-paraphrasing-detector

Proxy model: same as QQP