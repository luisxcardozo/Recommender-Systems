
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

The datasets used are: 
- ml-1m (MovieLens 1 million), composed by 1,000,209 anonymous ratings of roughly 3,706 movies by 6,040 users, ratings are contained in the file "ratings.dat" without header row.
- ml-20m (MovieLens 20 million), with 20,000,263 ratings of 26,744 movies by 138493 users, ratings are contained in the file "ratings.csv."

(In both cases, timestamps are represented in seconds starting midnight Coordinated Universal Time (UTC) of January 1, 1970. Each user has at least 20 ratings)

## 2. Determining benchmark.
To download the dataset, please install Pandas package first. Then issue the following command:

```
python ../datasets/movielens.py
```
Arguments:
 ```
- -data_dir: 
 ```
where to download and save the preprocessed data. By default, it is /tmp/movielens-data/.
```
--dataset: 
```
dataset name to be downloaded and preprocessed. By default, it is ml-1m.
Use the --help or -h flag to get a full list of possible arguments.

Note the ml-20m dataset is large (the rating file is ~500 MB), and it may take several minutes (~2 mins) for data preprocessing. Both the ml-1m and ml-20m datasets will be coerced into a common format when downloaded.

### To train and evaluate the model, issue the following command:
```
python ncf_main.py
```
Arguments:
```
--model_dir: ( default - /tmp/ncf/)
```
```
--data_dir: (This should be set to the same directory given to the data_download's data_dir argument.)
```
```
--dataset: (default - it is ml-1m)
```
For additional arguments for models and training process use the --help or -h flag to get a full list, with detailed descriptions.


