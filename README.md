
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
--data_dir: (default - /tmp/movielens-data/ 0
 ```
```
--dataset: (default - it is ml-1m)
```
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

## Step 3. Optimizing TensorFlow* for CPU.  
(*PERFORMANCE OPTIMIZATION*)
<img align="right" width="359" height="82" src="https://github.com/luisxcardozo/Image-Segmentation/blob/master/ISBackground/Step_three.PNG"> 
Intel developed specialized primitives libraries that increase Deep Neural Network model performance. This performance boost can be installed from Anaconda* or from the Intel® channel and run on Linux*, and on Windows* or OS*. 

- [Guide: Intel® Optimization for TensorFlow* Installation Guide](https://software.intel.com/en-us/articles/intel-optimization-for-tensorflow-installation-guide)

## NCF OPTIMIZATION

Run the launch_benchmark.py script with the appropriate parameters.
```
--model-source-dir - (path to official tensorflow model)
```
```
--checkpoint - (path to checkpoint directory for the Pre-trained model)
```
#### For Throughput:
- batch-size 256, 
- socket-id 0, 
- checkpoint, and
- model-source-dir
```
$ python launch_benchmark.py \
    --checkpoint /home/myuser/ncf_fp32_pretrained_model \
    --model-source-dir /home/myuser/tensorflow/models \
    --model-name ncf \
    --socket-id 0 \
    --batch-size 256 \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl
```
Tail of Throughput log:
```
...
2018-11-12 19:42:44.851050: step 22900, 931259.2 recommendations/sec, 0.27490 msec/batch
2018-11-12 19:42:44.880778: step 23000, 855571.2 recommendations/sec, 0.29922 msec/batch
2018-11-12 19:42:44.910551: step 23100, 870836.8 recommendations/sec, 0.29397 msec/batch
2018-11-12 19:42:44.940675: sE1112 19:42:45.420336 140101437536000 tf_logging.py:110] CRITICAL - Iteration 1: HR = 0.2248, NDCG = 0.1132
tep 23200, 867319.7 recommendations/sec, 0.29516 msec/batch
2018-11-12 19:42:44.971828: step 23300, 867319.7 recommendations/sec, 0.29516 msec/batch
2018-11-12 19:42:45.002699: step 23400, 861751.1 recommendations/sec, 0.29707 msec/batch
2018-11-12 19:42:45.033635: step 23500, 873671.1 recommendations/sec, 0.29302 msec/batch
Average recommendations/sec across 23594 steps: 903932.8 (0.28381 msec/batch)
...
```
#### Latency:
- batch-size 1, 
- socket-id 0, 
- checkpoint, and
- model-source-dir

```
$ python launch_benchmark.py \
    --checkpoint /home/myuser/ncf_fp32_pretrained_model \
    --model-source-dir /home/myuser/tensorflow/models \
    --model-name ncf \
    --socket-id 0 \
    --batch-size 1 \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl
```
Latency log:

```
...
2018-11-12 20:24:24.986641: step 6039100, 4629.5 recommendations/sec, 0.21601 msec/batch
2018-11-12 20:24:25.010239: step 6039200, 4369.1 recommendations/sec, 0.22888 msec/batch
2018-11-12 20:24:25.033854: step 6039300, 4583.9 recommendations/sec, 0.21815 msec/batch
2018-11-12 20:24:25.057516: step 6039400, 4696.9 recommendations/sec, 0.21291 msec/batch
2018-11-12 20:24:25.080979: step 6039500, 4788.0 recommendations/sec, 0.20885 msec/batch
2018-11-12 20:24:25.104498: step 6039600, 4405.8 recommendations/sec, 0.22697 msec/batch
2018-11-12 20:24:25.128331: step 6039700, 4364.5 recommendations/sec, 0.22912 msec/batch
2018-11-12 20:24:25.151892: step 6039800, 4485.9 recommendations/sec, 0.22292 msec/batch
2018-11-12 20:24:25.175342: step 6039900, 4675.9 recommendations/sec, 0.21386 msec/batch
2018-11-12 20:24:25.198717: step 6040000, 4905.6 recommendations/sec, 0.20385 msec/batch
Average recommendations/sec across 6040001 steps: 4573.0 (0.21920 msec/batch)
...
```
#### Accuracy
- batch-size 256,
- socket-id 0, 
- checkpoint path, and 
- model-source-dir
```
$ python launch_benchmark.py \
    --checkpoint /home/myuser/ncf_fp32_pretrained_model \
    --model-source-dir /home/myuser/tensorflow/models \
    --model-name ncf \
    --socket-id 0 \
    --accuracy-only \
    --batch-size 256 \
    --framework tensorflow \
    --precision fp32 \
    --mode inference \
    --docker-image intelaipg/intel-optimized-tensorflow:latest-devel-mkl
```
Log: (HR: Hit Ratio (HR) NDCG: Normalized Discounted Cumulative Gain)
```
...
E0104 20:03:50.940653 140470332344064 tf_logging.py:110] CRITICAL - Iteration 1: HR = 0.2290, NDCG = 0.1148
...
```
# RESULTS
Our engineers designed the elimination of inefficiencies in stages. Results shown in the following table.


| Optimization Step | Throughput (Image/sec) | Performance Improvement |
| :---         |     :---:      |    :---:      |
|Benchmark   | (ip)     |     |
| Optimized TensorFlow*     | (ip)       | tbd     |
| NCF Optimization  | **IP**      | **xxX**      |

# CONCLUSION
The optimization of TensorFlow* allows for deep-learning models built for this common framework to run several magnitudes faster on Intel® processors to increase scaling and analytical flexibility. The Xeon® processor is designed to scale comfortably to reduce training time of machine learning models. The collaboration between Intel® and Google* engineers to optimize TensorFlow* for higher performance on CPUs is part of ongoing efforts to increase the flexibility of AI applications by running on multiple mediums. Intel® believes the expansion of this accessibility is critical in the development of the next generation of AI models, and our efforts shed light into this by obtaining a projected 5.4x performance improvement with Intel® Xeon® Platinum 8180®. 



