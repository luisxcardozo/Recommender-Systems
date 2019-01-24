
# OPTIMIZATION OF NCF RECOMMENDATION SYSTEM FOR CPU IN TENSORFLOW

## BACKGROUND
Recommendation Systems are ranking systems that predict user responses to set options. 
The input normally an input query framed within a user and context specific information, and output a ranked list. These models are concerned with Memorization, or the learning of the frequent co-occurrence of items, and Generalization, that deals with the transitivity of these correlations.
These generalized linear models with nonlinear feature transformations are widely used for large-scale regression and classification
problems with sparse inputs. 

### Neural Collaborative Filtering
Neural Collaborative Filtering (NCF) method was published in 2017 by a combined effort by National University of Singapore, 
Columbia University, Shandong University, and Texas A&M University. It leverages neural networks to build the recommender system 
and proves that matrix factorization is a special case within NCF. It provides strong solutions and has been shown to outperform other state-of-the-art models in two public datasets. 
Neural Collaborative Filtering can be described as a generalization of Matrix Factorization.
In Recommender System, Matrix Factorization refers to the decomposition of the Utility Matrix into sub-matrices. During prediction, these sub-matrices are multiplied in an attempt to replicate the original Utility Matrix, which is factorized to minimize the error between the multiplication and the original matrix (Fig. 1).

![Matrix Vectorization](https://github.com/luisxcardozo/Recommender-Systems/blob/master/data/matrix%20vectorization.PNG)

But this dot product limits the expressiveness of the Item and User vectors.

## ARCHITECTURE
This framework adopts a multi-layer representation to model user interaction through a full neural treatment of collaborative treatment.
The bottom layer consist of two feature vectors VUu and VIi  that describe the user-item dynamic and can be customized to support a range of conditions to model users and items.

![NCF ARCHITECTURE](xxx)
