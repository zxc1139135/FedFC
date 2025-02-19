# FedFC

This is a lite version of codes for our paper entitled 
*FedFC: Accurate and Privacy-Preserving Vertical Federated Learning with Feature Conversion*.

## Environment
We conduct all experiments on the servers with Ubuntu 18.04 and Python 3.8.10 Please clone this repo and execute the following commands under the `FedFC` directory:

## Dataset Preparation
The datasets are expected to be downloaded manually in advance. 
Please download and unzip the [MNIST](http://yann.lecun.com/exdb/mnist/) 
and [CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html) datasets to the `dataset/mnist` and `dataset/cifar-10` directories, respectively. 


## Experiments
We prepare scripts for our experiments. For instance, you can run the FedFC training on the server accordingly:
```shell
# for example
$ python main_pipeline.py --gpu 1 --configs main_task/4_party/dp_01/cifar10_test_attack_dp_with_DT
```

