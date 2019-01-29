
# <p align="center">OPTIMIZATION OF RECOMMENDATION SYSTEM WITH NEURAL COLLABORATIVE FILTERING FOR CPU IN TENSORFLOW
<p align="center">Bhavani Subramanian<sup>1</sup>, Shankar Ratneshwaran<sup>2</sup>, Luis Cardozo<sup>2</sup>

- 1 - AI Deep Learning Algorithms - Oregon, Intel
- 2 - Artificial Intelligence Center For Excellence at Intel – Santa Clara, TCS

## GOAL
Learn CPU performance optimizations for recommendation systems with Neural Collaborative Filtering by:

- Optimizing Tensorflow* to run faster on CPU;
- Eliminating technology driven bottlenecks via thread optimizations;


### ABSTRACT  
2.4X performance improvement for inference on Intel® 8180 for recommender systems with Neural Collaborative Filtering, by reviewing bottleneck opportunities with thread-optimization analysis and Tensorflow* for CPU optimization. 

Models’ performance is improved by leveraging Intel®’s highly optimized math routines for deep learning. This primitives library is called Intel® Math Kernel Library for Deep Neural Networks (MKL-DNN) and includes convolution, normalization, activation and inner product, and other primitives. These steps are highly relevant as recent academic articles predict the development of non-static neural networks that increase memory and computational requirements, especially where accuracy minimization is paramount, like in the bio-med industry.


KEYWORDS. Convolutional Neural Networks, Neural Collaborative Filtering, Recommender Systems, Tensorflow Optimization,

### [BACKGROUND AND ARCHITECTURE](https://github.com/luisxcardozo/Recommender-Systems/blob/master/data/NCF_Bacground.md)

#### Evaluation Environment

|  |  | 
| :---         | :---        | 
|HW   | SKX Platinum 8180 CPU @ 2.50 GHz     |
| Tensorflow   | r1.10, commit: 958d5d0c6b22ca604363b3fc4547510bede3e3b1    |
| MKLDNN   | v0.16  |
| Dataset | MovieLens 1M ((http://files.grouplens.org/datasets/movielens/)) |


## 1. Getting started. 
(these instructions describe how to access the dataset and the process of installing the necessary prerequisites as well as running the NCF model)

Access and download the [MovieLen1Ms](http://files.grouplens.org/datasets/movielens/) datasets..

We will use the ML-1m dataset, but the datasets available are: 
- ml-1m (MovieLens 1 million), composed by 1,000,209 anonymous ratings of roughly 3,706 movies by 6,040 users, ratings are contained in the file "ratings.dat" without header row.
- ml-20m (MovieLens 20 million), with 20,000,263 ratings of 26,744 movies by 138493 users, ratings are contained in the file "ratings.csv."

(In both cases, timestamps are represented in seconds starting midnight Coordinated Universal Time (UTC) of January 1, 1970. Each user has at least 20 ratings)


 ### Install prerequisites:
* Python 2.7
* Follow instructions from https://github.com/NervanaSystems/tensorflow-models/tree/master/official#requirements for installing the requirements

 ### Clone repository:
```
  $ git clone https://github.com/NervanaSystems/tensorflow-models.git -b bhavanis/ncf
  $ cd tensorflow-models/
```

 ### Running inference:
```
  $ python run_tf_benchmark.py --checkpoint /mnt/aipg_tensorflow_shared/validation/dataset/q3models/ncf/ncf_trained_movielens_1m/ --data-location /mnt/aipg_tensorflow_shared/validation/dataset/q3models/ncf/ml-1m/ --single-socket --inference-only
```

## 2. Determining baseline.

[Instructions to train and evaluate model](https://github.com/tensorflow/models/tree/master/official/recommendation#train-and-evaluate-model) 

If you prefer to run the ml-20m dataset, note that it is large (the rating file is ~500 MB), and it may take several minutes (~2 mins) for data preprocessing. Both the ml-1m and ml-20m datasets will be coerced into a common format when downloaded.


## Step 3. Optimizing TensorFlow* for CPU.  
Optimize TensorFlow* for CPU.

Intel developed specialized primitives libraries that increase Deep Neural Network model performance. This performance boost can be installed from Anaconda* or from the Intel® channel and run on Linux*, and on Windows* or OS*. 

- [Guide: Intel® Optimization for TensorFlow* Installation Guide](https://software.intel.com/en-us/articles/intel-optimization-for-tensorflow-installation-guide)

## Step 4. NCF CORE OPTIMIZATION ANALYSIS RESULTS

1. BS-512, inter-op – 1, intra-op – 11, OMP_NUM_THREADS – 11**


### <p align="center">When OMP_NUM_THREADS was varied from 1 through 28, 11 yielded the best performance
![Thread_Iptimization](https://github.com/luisxcardozo/Recommender-Systems/blob/master/data/Thread_Optimization.png)

# RESULTS
Our engineers designed the elimination of inefficiencies in stages. Results shown in the following table.


| Cores | Throughput (recommendations/sec) | Performance Improvement |
| :---         |     :---:      |    :---:      |
|28 (default)   | 648,239.1     |    |
| 11     | 1,526,122       | **2.4X**     |


# CONCLUSION
The optimization of TensorFlow* allows for deep-learning models built for this common framework to run several magnitudes faster on Intel® processors to increase scaling and analytical flexibility. The Xeon® processor is designed to scale comfortably to reduce training time of machine learning models. The collaboration between Intel® and Google* engineers to optimize TensorFlow* for higher performance on CPUs is part of ongoing efforts to increase the flexibility of AI applications by running on multiple mediums. Intel® believes the expansion of this accessibility is critical in the development of the next generation of AI models, and our efforts shed light into this by obtaining increased performance with Intel® Xeon® Platinum 8180®. 



