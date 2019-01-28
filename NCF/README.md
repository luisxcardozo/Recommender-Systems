
# OPTIMIZATION OF RECOMMENDATION SYSTEM WITH NEURAL COLLABORATIVE FILTERING FOR CPU IN TENSORFLOW
Bhavani Subramanian<sup>1</sup>, Shankar Ratneshwaran<sup>2</sup>, Luis Cardozo<sup>2</sup>

- 1 - Developer Product Division Machine Learning Translator - Shangai, Intel
- 2 - Artificial Intelligence Center For Excellence at Intel – Santa Clara, TCS

## GOAL
This tutorial will introduce you to CPU performance optimizations for recommendation systems with Neural Collaborative Filtering and provide performance improvements by:

- Means to optimize Tensorflow* to run faster on CPU;
- Ways to eliminate technology driven bottlenecks via thread optimizations;


### ABSTRACT  
Tensorflow* CPU optimization for recommender systems with Neural Collaborative Filtering, on Intel® Xeon® processor-based platforms. performance improvement for training and inference on Intel® 8180 against an unoptimized run by solving bottlenecks with Neural Collaborative Filtering (NCF) framework with Neural Matrix Factorization (NeuMF) model as described in the Neural Collaborative Filtering paper. 

Models’ performance is improved by leveraging Intel®’s highly optimized math routines for deep learning. This primitives library is called Intel® Math Kernel Library for Deep Neural Networks (MKL-DNN) and includes convolution, normalization, activation and inner product, and other primitives, and by reviewing bottleneck opportunities with thread-optimization analysis. These steps are highly relevant as recent academic articles predict the development of non-static neural networks that increase memory and computational requirements, especially where accuracy minimization is paramount, like in the bio-med industry.


KEYWORDS. Convolutional Neural Networks, Neural Collaborative Filtering, Recommender Systems, Tensorflow Optimization,

### [BACKGROUND AND ARCHITECTURE](https://github.com/luisxcardozo/Recommender-Systems/blob/master/data/NCF_Bacground.md)

#### Evaluation Environment (*INCLUDE OBJECT*)

|  |  | 
| :---         | :---        | 
|HW   | SKX Platinum 8180 CPU @ 2.50 GHz     |
| Tensorflow   | r1.10, commit id: 958d5d0c6b22ca604363b3fc4547510bede3e3b1    |
| MKLDNN   | v0.16  |
| Dataset | MovieLens 1M ((http://files.grouplens.org/datasets/movielens/)) |


## 1. Getting started. 
(these instructions describe how to access the dataset and the process of installing the necessary prerequisites as well as running the NCF model)

Access and download the [MovieLen1Ms](http://files.grouplens.org/datasets/movielens/) datasets..

We will iuse the ML-1m dataset, but the datasets available are: 
- ml-1m (MovieLens 1 million), composed by 1,000,209 anonymous ratings of roughly 3,706 movies by 6,040 users, ratings are contained in the file "ratings.dat" without header row.
- ml-20m (MovieLens 20 million), with 20,000,263 ratings of 26,744 movies by 138493 users, ratings are contained in the file "ratings.csv."

(In both cases, timestamps are represented in seconds starting midnight Coordinated Universal Time (UTC) of January 1, 1970. Each user has at least 20 ratings)


  ## Install prerequisites:
* Python 2.7
* Follow instructions from https://github.com/NervanaSystems/tensorflow-models/tree/master/official#requirements for installing the requirements

  ## Clone repository:
```
  $ git clone https://github.com/NervanaSystems/tensorflow-models.git -b bhavanis/ncf
  $ cd tensorflow-models/
```

  ## Running inference:
```
  $ python run_tf_benchmark.py --checkpoint /mnt/aipg_tensorflow_shared/validation/dataset/q3models/ncf/ncf_trained_movielens_1m/ --data-location /mnt/aipg_tensorflow_shared/validation/dataset/q3models/ncf/ml-1m/ --single-socket --inference-only
```

## 2. Determining baseline.
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

If you prefer to run the ml-20m dataset, note that it is large (the rating file is ~500 MB), and it may take several minutes (~2 mins) for data preprocessing. Both the ml-1m and ml-20m datasets will be coerced into a common format when downloaded.


## Step 3. Optimizing TensorFlow* for CPU.  
(*PERFORMANCE OPTIMIZATION*)

Intel developed specialized primitives libraries that increase Deep Neural Network model performance. This performance boost can be installed from Anaconda* or from the Intel® channel and run on Linux*, and on Windows* or OS*. 

- [Guide: Intel® Optimization for TensorFlow* Installation Guide](https://software.intel.com/en-us/articles/intel-optimization-for-tensorflow-installation-guide)

## Step 4. NCF CORE OPTIMIZATION ANALYSIS

1. BS-512, inter-op – 1, intra-op – 11, OMP_NUM_THREADS – 11**
2. Timeline and VTune profiles unreliable
   - Overall time reported by timeline is greater (50x) than actual time
     - Timeline tool’s own overhead
   - VTune crashes
     - Also, AFAIK VTune uses a sampling frequency of 10ms
2. Approaches to determine hotspots:
   - Cycle accounting using SEP or a bigger dataset (ex. MovieLens-20m)
* (When OMP_NUM_THREADS was varied from 1 through 28, 11 yielded the best performance) *
![Thread_Iptimization](https://github.com/luisxcardozo/Recommender-Systems/blob/master/data/Thread_Optimization.png)
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
|Baseline   | (ip)     |     |
| Optimized TensorFlow*     | (ip)       | tbd     |
| NCF Optimization  | **IP**      | **xxX**      |

# CONCLUSION
The optimization of TensorFlow* allows for deep-learning models built for this common framework to run several magnitudes faster on Intel® processors to increase scaling and analytical flexibility. The Xeon® processor is designed to scale comfortably to reduce training time of machine learning models. The collaboration between Intel® and Google* engineers to optimize TensorFlow* for higher performance on CPUs is part of ongoing efforts to increase the flexibility of AI applications by running on multiple mediums. Intel® believes the expansion of this accessibility is critical in the development of the next generation of AI models, and our efforts shed light into this by obtaining a xx.xX performance improvement with Intel® Xeon® Platinum 8180®. 



