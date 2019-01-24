
# OPTIMIZATION OF RECOMMENDATION SYSTEM FOR CPU IN TENSORFLOW

## GOAL
This tutorial will introduce you to CPU performance optimizations for recommendation systems with Neural Collaborative Filtering and provide performance improvements by:

- Means to optimize Tensorflow* to run faster on CPU;
- Ways to eliminate technology driven bottlenecks with Neural Collaborative Filtering;


### ABSTRACT  
Tensorflow* CPU optimization for recommender systems with Neural Collaborative Filtering, on Intel® Xeon® processor-based platforms. xxxX improvement in performance for training on Intel® 8180 against an unoptimized run by solving bottlenecks with Neural Collaborative Filtering (NCF) framework with Neural Matrix Factorization (NeuMF) model as described in the Neural Collaborative Filtering paper. Current implementation is based on the code from the authors' NCF code and the Stanford implementation in the MLPerf Repo.

Models’ performance are optimized by leveraging Intel®’s highly optimized math routines for deep learning. This primitives library is called Intel® Math Kernel Library for Deep Neural Networks (MKL-DNN) and includes convolution, normalization, activation and inner product, and other primitives, and by reviewing bottleneck opportunities within the model’s sections. These steps are highly relevant as recent academic articles predict the development of non-static neural networks that increase memory and computational requirements, especially where accuracy minimization is paramount, like in the bio-med industry.


KEYWORDS. Convolutional Neural Networks, Neural Collaborative Filtering, Recommender Systems, Tensorflow Optimization,

### BACKGROUND AND ARCHITECTURE (*INCLUDE OBJECT*)

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
Access and download the [MovieLens](http://files.grouplens.org/datasets/movielens/) datasets, and prepare the data for training.

## 2. Determining benchmark.




