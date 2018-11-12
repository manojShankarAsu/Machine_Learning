import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets


def sigmoid(x):
	return 1/(1 + np.exp(-x))

def mylog(x):
	return np.log(x)

def oneminus(x):
	return 1-x

sig = np.vectorize(sigmoid)
log_v = np.vectorize(mylog)
oneminus_v = np.vectorize(oneminus)

def myplot(X,Y):
	index = 0
	for point in X:
		color = 'red'
		if Y[index] == 1:
			color = 'blue'
		plt.scatter(point[0],point[1],c=color)
		index +=1
	plt.show()


def cost(W,X,Y,m):
	h0 = W.dot(X)	
	h0 = sig(h0) # 1 * m
	log_h0  = log_v(h0)
	a = log_h0.dot(Y.T) # y.T = m * 1
	Y_1 = oneminus_v(Y) # 1 * m
	h0_1 = oneminus_v(h0) # 1 * m
	log_1_h0 = log_v(h0_1) # 1 * m
	b = log_1_h0.dot(Y_1.T) # 1 * m  X m * 1
	return (a+b) / m

def gradient(W,X,Y,m):
	h0 = W.dot(X)	
	h0 = sig(h0) # 1 * m
	A = h0 - Y # 1 * m
	G = A.dot(X.T) # 1 * m  X m * 3
	return G # 1 * 3


def prediction(W,X):
	h = W.dot(X)
	return sigmoid(h)


def learn_model(W,X,Y,m,learning_rate,no_of_iterations):
	iterations=[]
	cost_i=[]
	for i in range(no_of_iterations):
		if i % 1000 == 0:
			iterations.append(i)
			cost_i.append(cost(W,X,Y,m))
		G = gradient(W,X,Y,m)
		G = G * learning_rate
		G = G / m
		W = W - G
	plt.scatter(iterations,cost_i,c = 'blue',label='Scatter plot')
	plt.xlabel('iterations')
	plt.ylabel('Cost')	
	plt.show()
	return W

def pred_prob(x):
	if x >=0.5:
		return 1
	else:
		return 0 


def main():
	iris = datasets.load_iris()
	X = iris.data[:,:2]
	X = np.array(X)
	m = len(X)
	x0 = np.ones(m)
	X = np.array([x0,X[:,0],X[:,1]])
	print 'X'
	print X.shape
	Y = (iris.target != 0) * 1
	Y = np.array(Y)
	print Y
	WT = np.array([0,0,0])
	new_W = learn_model(WT,X,Y,m,0.1,300000)
	#new_W = np.array([-25.96818124,12.56179068,-13.44549335])
	print 'New W'
	print new_W
	print 'predictions'
	p =  prediction(new_W,X)
	pred_prob_v = np.vectorize(pred_prob)
	p = pred_prob_v(p)
	print p


	# print 'Cost'
	# print cost(WT,X,Y,m)
	# print 'gradient'
	# print gradient(WT,X,Y,m)
	#myplot(X,Y)
	# test = np.array([1,2,3])
	# v = np.vectorize(mylog)
	# res = v(test)
	# print 'res'
	# print res
	# print 'test'
	# print test
#https://medium.com/@martinpella/logistic-regression-from-scratch-in-python-124c5636b8ac
# started with learning rate of 0.001. 
# The cost was negative and slowly going up to 0
# I think we have negative cost since we are not squaring the cost
# lr = 0.1 and epochs = 300000 gives the fit parameter values
#






if __name__ == '__main__':
	main()