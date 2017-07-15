# IMPLEMENTATION OF SVM 

from sklearn.svm import SVC, LinearSVC
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import csv

#THE SVM FUNCTION IS IMPLEMENTED HERE 
def SVM(X, Y, type = 1, output_file="output.csv"):
	'''
		X_train - Training data
		Y_train - Corresponding training labels
		X_test - Test data
		Y_test - Corresponding test labels
		name - Name of the Model being implemented 
		parameters - parameters on which Grid Search has to be performed 
		svr - the type of classifier
		clf - trained classifier
		Y_train_pred - Predicted y labels after applying the learned classifier on training data
		Y_test_pred - Predicted y labels after applying the learned classifier on test data
		best_score - Accuracy of the classifier on the training data
		test_score - Accuracy of the classifier on the test data
	'''
	X_train, Y_train, X_test, Y_test = hold_out_divide(X, Y)
	X_train = normalize(X_train)
	X_test = normalize(X_test)
	name = ''

	#SVM using Linear Kernel with Grid Search on the C values
	if type==1:
		name = 'svm_linear'
		parameters = {'C':[0.1, 0.5, 1, 5, 10, 50, 100]}
		svr = SVC(kernel = 'linear')

	#SVM using Polynomial Kernel with Grid Search on C, degree and Gamma Values
	elif type==2:
		name = 'svm_polynomial'
		parameters = {'C':[0.1, 1, 3], 'degree':[4, 5, 6], 'gamma': [0.1, 1]}
		svr = SVC(kernel = 'poly')

	#SVM using RBF Kernel with Grid Search on C and Gamma values
	elif type==3:
		name = 'svm_rbf'
		parameters = {'C':[0.1, 0.5, 1, 5, 10, 50, 100], 'gamma': [0.1, 0.5, 1, 3, 6, 10]}
		svr = SVC(kernel = 'rbf')

	#Logistic Regression with Grid Search on C 	
	elif type==4:
		name = 'logistic'
		parameters = {'C':[0.1, 0.5, 1, 5, 10, 50, 100]}
		svr = LinearSVC()
	
	#k-Nearest Neighbours with Grid Search on n_neighbors and leaf_size 	
	elif type==5:
		name = 'knn'
		parameters = {'n_neighbors': range(1, 50), 'leaf_size': range(5, 60, 5)}
		svr = KNeighborsClassifier()
	
	#Decision Trees with Grid Search on Max Depth and Minimum Samples Split 	
	elif type==6:
		name = 'decision_tree'
		parameters = {'max_depth': range(1, 50), 'min_samples_split': range(2, 10, 1)}
		svr = DecisionTreeClassifier()
	
	#Random Forest with Grid Search on Max Depth and Minimum Samples Split 	
	else:
		name = 'random_forest'
		parameters = {'max_depth': range(1, 50), 'min_samples_split': range(2, 10, 1)}
		svr = RandomForestClassifier()


	#Training the classifier
	clf = GridSearchCV(svr, parameters, cv=5)
	clf.fit(X_train, Y_train)
	Y_train_pred = clf.predict(X_train)
	Y_test_pred = clf.predict(X_test)
	test_score = accuracy_score(Y_test_pred, Y_test)
	best_score = accuracy_score(Y_train_pred, Y_train)

	#Printing the values on the output csv file
	with open(output_file, 'a', newline = '') as csvfile:
		fieldnames = ['name', 'best_score', 'test_score']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writerow({'name': name, 'best_score':best_score, 'test_score': test_score})
	
	return "SUCCESS"


#STRATIFIED SAMPLING IS DONE HERE 
def get_stratified_split(X, Y):
	stratified_sampler = StratifiedShuffleSplit(n_splits = 1, test_size = 0.4, train_size = 0.6, random_state = 0)
	train_index = []
	test_index = []

	for t1_index, t2_index in stratified_sampler.split(X, Y):
		train_index = t1_index
		test_index = t2_index

	return train_index, test_index


#THE TRAINING-TEST HOLD-OUT-DIVIDE IS DONE HERE
def hold_out_divide(X, Y):
	train_index, test_index = get_stratified_split(X, Y)
	X_train = X[train_index] 
	Y_train = Y[train_index]
	X_test = X[test_index]
	Y_test = Y[test_index]

	return X_train, Y_train, X_test, Y_test


#THE TRAINING DATA IS NORMALIZED HERE
def normalize(X):
	mu = np.mean(X, axis=0)
	std = np.std(X, axis=0)
	X = (X - mu)/std
	return X
