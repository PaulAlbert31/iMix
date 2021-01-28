# Unofficial Pytorch implementation for *iMix: A Strategy for regularizing Contrastive Representation Learning*.
Paper: https://openreview.net/pdf?id=T6AxtOaWydQ


## Requirements
The environment.yml file contains the required packages. You can install them in a new conda environment as follows:
```
conda env create -f environment.yml
conda activate iMix
```

## Run iMix+N-pairs
Run an experiment for cifar100 on GPU0:
```
CUDA_VISIBLE_DEVICES=0 python main.py --save-dir CIFAR100_resnet18 --net resnet18 --dataset cifar100
```
Resume training
```
CUDA_VISIBLE_DEVICES=0 python main.py --save-dir CIFAR100_resnet18 --net resnet18 --dataset cifar100 --resume CIFAR100_resnet18/last_model.pth.tar
```

Multi-gpu is supported using the [torch.nn.DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html) module.

Multiple GPUs on cifar10 using a ResNet50:
```
CUDA_VISIBLE_DEVICES=0,1 python main.py --save-dir CIFAR10_resnet50 --net resnet50 --dataset cifar10
```
## Evaluate results using
### kNN
```
CUDA_VISIBLE_DEVICES=0 python eval_knn.py CIFAR10_resnet50
```
### Linear finetuning
```
CUDA_VISIBLE_DEVICES=0 python eval_linear.py CIFAR10_resnet50
```

## Results
| Dataset | Network | kNN | linear |
| -----------|-----------|------| ---- |
|CIFAR10|WideResNet28-2|
|       |ResNet18|
|       |ResNet50|
|CIFAR100|WideResNet28-2|
|        |ResNet18|
|        |ResNet50|
 
## Experiment using your own ... 
### Dataset
Define you dataset in the dataset folder using the [pytorch templates](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html) and the existing files in the [datasets](https://github.com/PaulAlbert31/iMix/tree/main/datasets) folder and add a call in the [`__init__.py`](https://github.com/PaulAlbert31/iMix/blob/main/datasets/__init__.py) file. 
### Network
Add you network in the net folder and call it at the beginning of [`main.py`](https://github.com/PaulAlbert31/iMix/blob/main/main.py). Don't forget to add a non-linear projection layer for optimal results

StackEdit stores your files in your browser, which means all your files are automatically saved locally and are accessible **offline!**

## Cite the original paper
```
@inproceedings{2021_ICLR_iMix,
  title="{i-Mix: A Strategy for Regularizing Contrastive Representation Learning}",
  author="Lee, Kibok and Zhu, Yian and Sohn, Kihyuk and Li, Chun-Liang and Shin, Jinwoo and Lee, Honglak",
  booktitle="{International Conference on Learning Representations (ICLR)}",
  year="2021"}
```
