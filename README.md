# Datasets

## Company

Description: This dataset is proposed at [SIGMOD'18](https://dl.acm.org/doi/10.1145/3183713.3196926) and recently studied by [Ditto](https://arxiv.org/abs/2004.00584) with LLMs.

Task: find the all the matching entities between two datasets

Dataset size:

| Table A | Table B | pairs |
|-|-|-|
|28.2K|28.2K|79.524M|

Oracle model: RoBerTa (check [Ditto's code](https://github.com/megagonlabs/ditto) for training)

Proxy: string similarity measure (check [AnHai's Group](https://sites.google.com/site/anhaidgroup/current-projects/magellan/py_stringmatching?authuser=0))

## Quora Question Pair (QQP)

Description: This dataset is released by Quora.com and recently studied by [SMART](https://aclanthology.org/2020.acl-main.197/)

Dataset size:
| Table | pairs |
|-|-|
|8M|8MM|

Oracle model: SMART_BERT (claimed to be released at [github repo](https://github.com/namisan/mt-dnn))

Proxy:
1. BiMPM (no pretrained released, need to train by ourselfes, scripts released at the [github repo](https://github.com/zhiguowang/BiMPM) )
2. Siamese-CNN (no pretrained, train script released, need to construct architeccture and train)
3. Pair-CNN [paper](https://dl-acm-org.proxy2.library.illinois.edu/doi/abs/10.1145/2766462.2767738)

## Stack Overflow

Description: This dataset is an offocial dump released by [MSR2013](http://2013.msrconf.org/challenge.php#challenge_data), where the duplication issue is [studied](https://link.springer.com/article/10.1007/s11390-015-1576-4).

Dataset size:
| Table | pairs |
|-|-|
| 10M | 100MM |

Oracle model: try to use pretrained BERT

Proxy: try to use the same as QQP

## ArXiv

Description: to be added

Dataset size: to be added

Oracle model: fine-tune Roberta from https://huggingface.co/AMHR/adversarial-paraphrasing-detector

Proxy model:
1. [Multiple-instance Learning Paraphrase (MultiP) Model](https://aclanthology.org/Q14-1034/) with training scripts (https://github.com/cocoxu/multip).
2. Logistic regression model, [training scripts](https://github.com/cocoxu/SemEval-PIT2015) implemented by [SemEval](https://aclanthology.org/S15-2001/).

## LanguageNet

Description: [LanguageNet](https://languagenet.github.io/) is a collection of sentence level paraphrases from Twitter by linking tweets through shared URLs.

Dataset size:
| Table 1 | Table 2 | pairs |
|-|-|-|
|51K|51K|2B|

Oracle model & proxy model: same as ArXiv

## Carla
