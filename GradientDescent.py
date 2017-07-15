# IMPLEMENTATION OF THE GRADIENT DESCENT ALGORITHM

import numpy as np 
import csv


#THE GRADIENT DESCENT ALGORITHM IS IMPLEMENTED HERE
def GradientDescent(X, Y, output_file):
	'''
		n - Number of rows in the training data
		d - Number of features in a training data
		X - Training data with a column of 1's added to it 
		Y - Labels of the training data 
		threshold - Maximum permissible deviation of beta's between two iterations 
		learning_rate - List of learning rates which is to be experimented 
		alpha - Learning rate 
		beta - weights 
		dR - Differential of the Loss function 
		f_X - Affine function which is evaluated 
		iter - Keeps track of the number of iterations 

	'''
	data = X
	X = normalize(X)
	X = np.insert(X, obj = 0, values = 1, axis = 1)
	threshold = 0.001
	n = X.shape[0]
	d = X.shape[1]
	Y = Y.reshape(n,1)
	learning_rate = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 0.9] 

	#Various learning-rates are experimented here 
	for alpha in learning_rate:
		iter = 0
		beta = np.zeros([d, 1])

		#Update beta until convergence
		while iter < 100:
			beta_prev = beta
			f_X = np.dot(X , beta)
			dR = 1/n* np.dot(np.transpose(X), (f_X-Y))
			beta = beta - alpha * dR
			
			deviation = np.linalg.norm(beta - beta_prev, ord = 1)	

			if deviation < threshold or iter > 100:
				break
			
			iter = iter + 1

		#Printing the values to the output csv file
		with open(output_file, 'a', newline = '') as csvfile:
			fieldnames = ['alpha', 'iterations', 'beta_intercept', 'beta_age', 'beta_weight']
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
			writer.writerow({'alpha': alpha, 'iterations':iter, 'beta_intercept': float(beta[0]), 'beta_age': float(beta[1]), 'beta_weight': float(beta[2])})


	return "SUCCESS"



#NORMALIZES THE INPUT TRAINING DATA
def normalize(X):
	mu = np.mean(X, axis=0)
	std = np.std(X, axis=0)
	X = (X - mu)/std
	return X


#VISUALIZING THE DATA AND BETA 
def visualize(X, Y, beta):
	import matplotlib
	import matplotlib.pyplot as plt
	from matplotlib import cm
	from mpl_toolkits.mplot3d import Axes3D 

	fig = plt.figure()
	ax = fig.gca( projection='3d')
	age = X[:,0]
	weight = X[:,1]
	height = Y
	age_grid, weight_grid = np.meshgrid(age, weight)
	#ax.plot_surface(age, weight, )
	ax.plot_surface(age_grid, weight_grid, beta[1]*age + beta[2]*weight + beta[0], rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
	ax.scatter(age, weight, height, c='red')
	ax.set_xlabel('Age(Years)')
	ax.set_ylabel('Weight(Kilograms)')
	ax.set_zlabel('Height(Meters)')

	plt.show()
	