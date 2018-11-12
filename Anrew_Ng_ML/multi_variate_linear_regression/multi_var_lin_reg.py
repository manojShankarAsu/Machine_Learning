import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20.0, 10.0)
from mpl_toolkits.mplot3d import Axes3D


def normalize(vector):
	mean = np.mean(vector)
	min_v = np.min(vector)
	max_v = np.max(vector)
	range_v = max_v - min_v
	arr = np.array([(x-mean)/range_v for x in vector])
	return arr

def mean_sq_error(W,X,Y,m):
	pred = prediction(W,X)
	error = pred - Y
	error = np.square(error)
	mse = np.sum(error) / m	
	return mse

def gradient(feature_j,W,X,Y,m):
	pred = prediction(W,X)
	error = pred - Y
	features_values = X[feature_j,:]
	total = error.dot(features_values.T)
	return total / m

def learn_model(W,X,Y,m,learning_rate,no_of_iterations):
	iterations=[]
	cost = []
	print 'Initial MSE'
	print mean_sq_error(W,X,Y,m)
	for i in range(no_of_iterations):
		tmp0 = 0
		tmp1 = 0
		tmp2 = 0
		tmp0 = gradient(0,W,X,Y,m)
		tmp1 = gradient(1,W,X,Y,m)
		tmp1 = gradient(2,W,X,Y,m)
		if i % 1000 == 0:		
			# print tmp0, tmp1, tmp2
			# print 'Cost'
			# print mean_sq_error(W,X,Y,m)
			# print '----'
			iterations.append(i)
			cost.append(mean_sq_error(W,X,Y,m))
		grad_vec = np.array([tmp0,tmp1,tmp2])
		grad_vec = grad_vec * learning_rate
		W = W - grad_vec
	plt.scatter(iterations,cost,c = 'blue',label='Scatter plot')
	plt.xlabel('iterations')
	plt.ylabel('Cost')	
	plt.show()
	return W




def prediction(W, X):
	pred = W.dot(X)
	return pred

def main():
	data = pd.read_csv('student.csv')
	#print data.shape
	#print data.head()	
	math = data['Math'].values
	#math = normalize(math)
	read = data['Reading'].values
	#read = normalize(read)
	writing = data['Writing'].values
	writing_mean = np.mean(writing)
	#writing = normalize(writing)
	m = len(math)
	# fig = plt.figure()
	# ax = Axes3D(fig)
	# y = 0.5 * math + 0.1	
	# ax.plot(math,y,color='blue',label='line')
	# ax.scatter(math,read,writing,color='red')
	# plt.show()
	m = len(math)
	x0 = np.ones(m)
	X = np.array([x0,math,read])
	#print 'Samples'
	#print X[0:3,0:5]
	#print X.shape
	WT = np.array([0,0,0])
	#print 'Targets'
	Y = np.array(writing)
	#print Y[0:5]
	#print 'Predictions'
	pred = prediction(WT,X)
	print 'Mean Square Error'
	print mean_sq_error(WT,X,Y,m)
	no_of_iterations = 50000
	learning_rate = 0.0001
	new_W = learn_model(WT,X,Y,m,learning_rate,no_of_iterations)
	print 'New W'
	print new_W
	print 'After Mean Square Error'
	print mean_sq_error(new_W,X,Y,m)
	fig = plt.figure()
	ax = Axes3D(fig)
	preds = new_W.dot(X)
	ax.plot(math,read,preds,color='blue',label='line')
	ax.scatter(math,read,writing,color='red')
	plt.show()
	new_data = np.array([1,90,90])
	print 'Prediction: for', new_data
	print prediction(new_W,new_data.T)




if __name__ == '__main__':
	main()

#The gradient descent started increasing after certain no of iterations
# Have to check why gradient descent increased

# I used learning_rate of 0.01 . This caused the model to overshoot the local minima and hence cost 
# kept increasing. I later tried with learning_rate of 0.0001 and 500000 iterations. 
# Now the model has good prediction and low error

# Without normalizing the prediction is better
# learning rate = 0.0001 and epochs = 50000