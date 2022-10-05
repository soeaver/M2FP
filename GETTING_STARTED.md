## Getting Started with M2FP

This document provides a brief intro of the usage of M2FP.

Please see [Getting Started with Detectron2](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md) for full usage.


### Training & Evaluation in Command Line

We provide a script `train.sh`, that is made to train all the configs provided in M2FP.
```
sh train.sh
```
The configs are made for 8-GPU training.


To evaluate a model's performance, use
```
sh test.sh
```
For more options, see `python train_net.py -h`.

