
# OPTIMIZATION OF WIDE AND DEEP RECOMMENDATION SYSTEM FOR CPU IN TENSORFLOW

## BACKGROUND
Recommendation Systems are ranking systems that predict user responses to set options. The input normally an input query framed within a user and context specific information, and output a ranked list. These models are concerned with Memorization, or the learning of the frequent co-occurrence of items, and Generalization, that deals with the transitivity of these correlations.
These generalized linear models with nonlinear feature transformations are widely used for large-scale regression and classification problems with sparse inputs. 

### Wide and Deep
One challenge in recommender systems, is to achieve both memorization and generalization. The former may be defined as learning the frequent co-occurrence of items and exploiting historical correlations and is a generalized linear model of the form y = wT x + b. The latter, is based on a transitivity of correlations that studies feature combinations that have not, or rarely, occurred, and is composed of a forward-fed neural network. Memorization Recommendations are typically more relevant to the studied features, whereas Generalization Recommendations improve the diversity of the recommended features.
The combination of the linear model (Wide) and neural network (Deep) allows to achieve both memorization and generalization in one model. This is performed with the weighted sum of their output log odds as their predictions. This prediction feeds one common logistic loss function for joint training. This training is performed by a backpropagation of the gradients from the output to both the wide and deep parts of the model simultaneously using mini-batch stochastic optimization.

## ARCHITECTURE

![ARCHITECTURE](https://github.com/luisxcardozo/Recommender-Systems/blob/master/data/WDArchitecture.PNG)
