# AI - ML Assignment #

## Introduction ##

In this assignment, Machine Learning algorithms like Perceptron, Gradient Descent and SVM are implemented. Functions from the scikit-learn library are implemented. 


### Perceptron Algorithm ###

In this problem, the perceptron learning algorithm is implemented on a linearly separable dataset. The input file considered for this problem consists of two features, representing the x and y coordinates of various points. The output label takes on a value of positive or negative one. The label can be thought of as separating the points into two categories. The weights corresponding to the features and the bias for every iteration is written to the output file. 


### Gradient Descent ###

In this problem, linear regression using gradient descent is applied to the input dataset. The input file consists of age (years) and weight (kg) as features. The output label is the height (meters). The data is preprocessed by scaling every feature by its standard deviation and setting its mean to zero. The gradient descent algorithm is applied with various learning rates {0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10}. For each value of the learning rate, the algorithm runs for 100 iterations. The learning rate, corresponding number of iterations for convergence, bias and weights corresponding to the features: age and weight are written to the output file in that order. 


### SVM ###

In this problem, various classification models are used on a chessboard-like input dataset. The input data is split into training (60%) and testing (40%). Stratified sampling is used to ensure that same ratio of positive and negative examples exist in both the training and testing samples. Cross-validation (with k = 5) is used instead of validation set. The following models are used for classification:

SVM with Linear Kernel 

SVM with Polynomial Kernel

SVM with RBF Kernel 

Logistic Regression

K-Nearest Neighbours 

Decision Trees 

Random Forest 



The method name, best training score and test scores are written to the output file 



## Code Description ##

problem1 : Takes  the linearly separable dataset as the input, applies the perceptron algorithm in it and writes the output to the output file 

problem2 : Takes an input dataset, applies Gradient Descent to it and writes the output to the output file

problem3 : Takes an input dataset, applies SVM Models with different kernels and writes the output to the output file

Perceptron : Perceptron algorithm implementation

GradientDescent : Gradient Descent implementation

SVMModels : SVM Models implementations. Kernels used: Linear, Polynomial, RBF, etc. 


## How to execute? ##

Run python <problem_name> <input_file> <output_file_name>

If you want to execute problem-1 on the file input1, you will run 

python problem1_3.py input/input1.csv output1.csv 


Similar steps would be followed for other problems as well
Note that sample input files are stored in the input folder 


## Result ## 

For gradient descent, it is observed that when the learning rate alpha is either too small or too large, the value of beta (weight) keeps fluctuating and hence, number of iterations is more. The optimum value of alpha was 1. The correct value of alpha must update beta such that the next beta is not too large than the first but also not too close to it to differentiate it from the next
Based on these observations, the value of 0.9 was chosen for alpha as it has the least number of iterations as 13
