# Deep Neural Matching Models for Graph Retrieval 
This paper has been accpeted by 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval.

by Kunal Goyal, Utkarsh Gupta, Abir De and Soumen Chakrabarti 


### Citation
```
@inproceedings{10.1145/3397271.3401216,
author = {Goyal, Kunal and Gupta, Utkarsh and De, Abir and Chakrabarti, Soumen},
title = {Deep Neural Matching Models for Graph Retrieval},
year = {2020},
isbn = {9781450380164},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3397271.3401216},
doi = {10.1145/3397271.3401216},
abstract = {Graph retrieval from a large corpus of graphs has a wide variety of applications, e.g., sentence retrieval using words and dependency parse trees for question answering, image retrieval using scene graphs, and molecule discovery from a set of existing molecular graphs. In such graph search applications, nodes, edges and associated features bear distinctive physical significance. Therefore, a unified, trainable search model that efficiently returns corpus graphs that are highly relevant to a query graph has immense potential impact. In this paper, we present an effective, feature and structure-aware, end-to-end trainable neural match scoring system for graphs. We achieve this by constructing the product graph between the query and a candidate graph in the corpus, and then conduct a family of random walks on the product graph, which are then aggregated into the match score, using a network whose parameters can be trained. Experiments show the efficacy of our method, compared to competitive baseline approaches.},
booktitle = {Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
pages = {1701–1704},
numpages = {4},
keywords = {product graph, random walk, scoring subgraph match, graph search},
location = {Virtual Event, China},
series = {SIGIR '20}
}
```

### Abstract
Graph retrieval from a large corpus of graphs has a wide variety of applications, e.g., sentence retrieval using words and dependency parse trees for question answering, image retrieval using scene graphs, and molecule discovery from a set of existing molecular graphs.  In such graph search applications, nodes, edges and associated features bear distinctive physical significance.  Therefore, a unified, trainable search model that efficiently returns corpus graphs that are highly relevant to a query graph has immense potential impact. In this paper, we present an effective, feature and structure-aware, end-to-end trainable neural match scoring system for graphs.  We achieve this by constructing the product graph between the query and a candidate graph in the corpus, and then conduct a family of random walks on the product graph, which are then aggregated into the match score, using a network whose parameters can be trained.  Experiments show the efficacy of our method, compared to competitive baseline approaches.
## Contents

- [Requirements](#requirements)
- [Pretrained Models](#pretrained-models)
- [Training and Inference](#training-and-inference)



## Requirements
1. Python 3.6
2. tensorflow 1.15.0
3. `pip install -r requirements.txt`
#
# Pretrained Models
Checkpoints are available in checkpoints/

## Datasets

You have to download the [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) dataset.

#### VQA Dataset Structure

```bash
python vaq_dataset_gen.py --data_path $CLEVR_{NOISE} --noise $NOISE
```

Also available in data/clevr/

#### SQuAD Dataset

```bash
python squaddep.py --squad ./data/squad/train-v2.0.json --data ./data/squad/squad.pkl
```

available in data/squad

## Training and Inference

#### 1 Train GxNet on Clevr dataset
```bash
python train.py --data_path ./data/clevr/clevr_{$NOISE}.pkl --name clevr_{$NOISE} --dataset clevr --logfile clevr_noise_{$NOISE}.log --num_queries 50 --num_walks 15 --max_length_walk 15 --sparse_walk True --walk_method random

```
#### 2 Train GxNet on SQuAD dataset
```bash
python train.py --data_path ./data/squad/squad.pkl --name squad --dataset squad --num_queries_train 1000 --num_queries_eval 100 --logfile ./logs/squad.log --early_stopping 10 --delta 0.2 --num_walks 16 --max_length_walk 16 --nlayer1 32 --nlayer2 32 --elayer1 8 --elayer2 8 --nelayer1 32 --nelayer2 32 --walk_method random --sparse_walk True
```
