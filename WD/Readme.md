
# OPTIMIZATION OF RECOMMENDATION SYSTEM WITH WIDE AND DEEP METHOD FOR CPU IN TENSORFLOW

## GOAL
This tutorial will introduce you to CPU performance optimizations for recommendation systems with Wide And Deep and provide performance improvements by:

- Means to optimize Tensorflow* to run faster on CPU;
- Ways to eliminate technology driven bottlenecks with Wide and Deep.


### ABSTRACT  
Tensorflow* CPU optimization for recommender systems with Wide and Deep, on Intel® Xeon® processor-based platforms. xxxX improvement in performance for training on Intel® 8180 against an unoptimized run.
Models’ performance are optimized by leveraging Intel®’s highly optimized math routines for deep learning. This primitives library is called Intel® Math Kernel Library for Deep Neural Networks (MKL-DNN) and includes convolution, normalization, activation and inner product, and other primitives, and by reviewing bottleneck opportunities within the model’s sections. These steps are highly relevant as recent academic articles predict the development of non-static neural networks that increase memory and computational requirements, especially where accuracy minimization is paramount, like in the bio-med industry.


KEYWORDS. Convolutional Neural Networks, Wide And Deep, Recommender Systems, Tensorflow Optimization,

### BACKGROUND AND ARCHITECTURE(*INCLUDE OBJECT*)

#### Evaluation Environment (*INCLUDE OBJECT*)

|  |  | 
| :---         | :---        | 
|HW   | Xeon Platinum 8180, @2.5G Turbo on, HT on, NUMA     |
| OS    | CentOS Linux 7 (Core)  kernel 3.10.0-693.el7.x86_64       |
| Tensorflow   | v1.8rc1, commit id: 2dc7357    |
| Keras  | v2.1.5      |
| MKLDNN   | v0.13  |
| Model	3D-UNet | (https://github.com/ellisdg/3DUnetCNN ) |
| Dataset | BraTS ((http://www.med.upenn.edu/sbia/brats2017.html)) |
| CMD (inference)| $python predict.py|
| BS | 1 |


## 1. Getting started.
Access and download the [Census Income]( https://archive.ics.uci.edu/ml/datasets/Census+Income) dataset, and prepare the data for training.

Download the following files: 
- adult.data, and 
- adult.test

## 2. Determining benchmark.

### For latency mode with: --batch-size = 1
 ```
 $ cd /home/myuser/models/benchmarks

$ python launch_benchmark.py \ 
      --framework tensorflow \ 
      --model-source-dir /home/myuser/path/to/tensorflow-models \
      --precision fp32 \
      --mode inference \
      --model-name wide_deep \
      --batch-size 1 \
      --data-location /home/myuser/path/to/dataset \
      --checkpoint /home/myuser/path/to/checkpoint \
      --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl \
      --verbose
 ```
### For throughput mode with: --batch-size = 1024
 ```
 $ python launch_benchmark.py \ 
      --framework tensorflow \ 
      --model-source-dir /home/myuser/path/to/tensorflow-models \
      --precision fp32 \
      --mode inference \
      --model-name wide_deep \
      --batch-size 1024 \
      --data-location /home/myuser/path/to/dataset \
      --checkpoint /home/myuser/path/to/checkpoint \
      --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl \
      --verbose
 ``` 
### Log file is saved to: models/benchmarks/common/tensorflow/logs

### Tail of log
 ``` 
accuracy: 1.0
accuracy_baseline: 1.0
auc: 1.0
auc_precision_recall: 0.0
average_loss: 2.1470942e-05
global_step: 9775
label/mean: 0.0
loss: 2.1470942e-05
precision: 0.0
prediction/mean: 2.1461743e-05
recall: 0.0
End-to-End duration is %s 36.5971579552
Latency is: %s 0.00224784460139
lscpu_path_cmd = command -v lscpu
lscpu located here: /usr/bin/lscpu
current path: /workspace/benchmarks
search path: /workspace/benchmarks/*/tensorflow/wide_deep/inference/fp32/model_init.py
Using model init: /workspace/benchmarks/classification/tensorflow/wide_deep/inference/fp32/model_init.py
PYTHONPATH: :/workspace/models
RUNCMD: python common/tensorflow/run_tf_benchmark.py         --framework=tensorflow         --model-name=wide_deep         --precision=fp32         --mode=inference         --model-source-dir=/workspace/models         --intelai-models=/workspace/intelai_models         --batch-size=1                  --data-location=/dataset         --checkpoint=/checkpoints
 ``` 
