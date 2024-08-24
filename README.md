# FLEXTAF: Enhancing Table Reasoning with Flexible Tabular Formats

This repository contains code for the paper ["FLEXTAF: Enhancing Table Reasoning with Flexible Tabular Formats"](https://arxiv.org/abs/2408.08841).

If you use FLEXTAF in your work, please cite it as follows:
```
@article{zhang2024flextaf,
  title={FLEXTAF: Enhancing Table Reasoning with Flexible Tabular Formats},
  author={Zhang, Xuanliang and Wang, Dingzirui and Dou, Longxu and Wang, Baoxin and Wu, Dayong and Zhu, Qingfu and Che, Wanxiang},
  journal={arXiv preprint arXiv:2408.08841},
  year={2024}
}
```

## Build Environment
```
conda create -n flex python=3.10
conda activate flex
pip install -r requirements.txt
```

## Pre-Process Data
Download and put each dataset in ./dataset, and run [dataset/slurm/preprocess.slurm](./dataset/slurm/preprocess.slurm).

## Download Model
Download the models and put them in ./model.

## FLEXTAF-Vote

### Reasoning
Run the table reasoning with [reason/slurm/inference.slurm](./reason/slurm/inference.slurm), in which can select the tabular format.

### Vote
Ensemble the results of multiple formats with [reason/slurm/vote.slurm](./reason/slurm/vote.slurm)

## FLEXTAF-Single

### Classification
This step is to train the classifier to predict the most suitable tabular format.

#### Obtain training data
Firstly run the table reasoning on the training set to get the training data with [reason/slurm/inference.slurm](./reason/slurm/inference.slurm) and [reason/slurm/vote.slurm](./reason/slurm/vote.slurm).

#### Train
Train the classifier with [classify/slurm/multi_label_finetune.slurm](./classify/slurm/multi_label_finetune.slurm).

## Reasoning
Predict the suitable tabular format with [classify/slurm/multi_label_classify.slurm](./classify/slurm/multi_label_classify.slurm). If with results of all candidate tabular formats, the performance of Flex-Formats-Single can be also obtained at this step.
