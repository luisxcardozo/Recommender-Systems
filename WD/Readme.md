
#  <p align="center">OPTIMIZATION OF RECOMMENDATION SYSTEM WITH WIDE AND DEEP METHOD FOR CPU IN TENSORFLOW
<p align="center">Wei Wang<sup>1</sup>, Shankar Ratneshwaran<sup>2</sup>, Kevin Bryan<sup>2</sup>, Luis Cardozo<sup>2</sup>

- 1 - Department - Location, Intel
- 2 - Artificial Intelligence Center For Excellence at Intel – Santa Clara, TCS



## GOAL
Learn CPU performance optimizations for income predictions with Wide & Deep optimizations by:
1.	Optimizing Tensorflow* to run faster on CPU;
2.	Eliminating technology driven bottlenecks.

### ABSTRACT

XX.Xx improvement on inference performance on Intel® 8180 by solving architecture bottlenecks by leveraging Intel®’s highly optimized math routines for deep learning and Tensorflow* CPU optimization.

Intel®’s primitives library is called Intel® Math Kernel Library for Deep Neural Networks (MKL-DNN) and includes convolution, normalization, activation and inner product, and other primitives, and by reviewing bottleneck opportunities within the model’s sections. These steps are highly relevant as recent academic articles predict the development of non-static neural networks that increase memory and computational requirements.


KEYWORDS. Convolutional Neural Networks, Wide And Deep, Recommender Systems, Tensorflow Optimization,

### [BACKGROUND AND ARCHITECTURE](https://github.com/luisxcardozo/Recommender-Systems/blob/master/data/WD_Background.md)



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


## Step 1. Getting started.
[download models and datasets (*with various applications and tutorials*)](https://github.com/tensorflow/models/tree/master/official/wide_deep)

Download the following files: 
- adult.data, and 
- adult.test

## Step 2. Optimizing TensorFlow* for CPU.  
(*PERFORMANCE OPTIMIZATION*)
<img align="right" width="359" height="82" src="https://github.com/luisxcardozo/Image-Segmentation/blob/master/ISBackground/Step_three.PNG"> 
Intel developed specialized primitives libraries that increase Deep Neural Network model performance. This performance boost can be installed from Anaconda* or from the Intel® channel and run on Linux*, and on Windows* or OS*. 

- [Guide: Intel® Optimization for TensorFlow* Installation Guide](https://software.intel.com/en-us/articles/intel-optimization-for-tensorflow-installation-guide)

## Step 3. WD OPTIMIZATION

(ip)

# RESULTS
Our engineers designed the elimination of inefficiencies in stages. Results shown in the following table.


| Optimization Step | Throughput (Image/sec) | Performance Improvement |
| :---         |     :---:      |    :---:      |
|Baseline   | (ip)     |     |
| Optimized TensorFlow*     | (ip)       | tbd     |
| WD Optimization  | **IP**      | **xxX**      |

# CONCLUSION
The optimization of TensorFlow* allows for deep-learning models built for this common framework to run several magnitudes faster on Intel® processors to increase scaling and analytical flexibility. The Xeon® processor is designed to scale comfortably to reduce training time of machine learning models. The collaboration between Intel® and Google* engineers to optimize TensorFlow* for higher performance on CPUs is part of ongoing efforts to increase the flexibility of AI applications by running on multiple mediums. Intel® believes the expansion of this accessibility is critical in the development of the next generation of AI models, and our efforts shed light into this by obtaining a xx.xX performance improvement with Intel® Xeon® Platinum 8180®. 

