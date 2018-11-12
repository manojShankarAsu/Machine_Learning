from matplotlib import pyplot as plt
import numpy as np 
import pandas as pd 


def prediction(w1,b,x):
	return w1 * x + b


def gradient_w1(head_size,brain_weight,w1,b):
	gradient = 0
	n = len(head_size)
	for i in range(n):
		gradient += (( prediction(w1,b,head_size[i]) - brain_weight[i]) * head_size[i])
	return gradient/n

def gradient_b(head_size,brain_weight,w1,b):
	gradient = 0
	n = len(head_size)
	for i in range(n):
		gradient += ( prediction(w1,b,head_size[i]) - brain_weight[i])
	return gradient/n


def mean_square_error(head_size,brain_weight,w1,b):
	n = len(head_size)
	error = 0
	for i in range(n):
		error +=  (prediction(w1,b,head_size[i]) - brain_weight[i]) ** 2
	return error / n 


def learn_model(w1,b,head_size,brain_weight,no_of_iterations,learning_rate_w1, learning_rate_b):
	tmp1 = 0
	tmpb = 0
	iterations=[]
	cost = []
	for i in range(no_of_iterations):		
		if i % 500 == 0:
			iterations.append(i)
			cost.append(mean_square_error(head_size,brain_weight,w1,b))
		tmp1 = w1 - learning_rate_w1 * gradient_w1(head_size,brain_weight,w1,b)
		tmpb = b - learning_rate_b * gradient_b(head_size,brain_weight,w1,b)
		w1 = tmp1
		b = tmpb
	plt.scatter(iterations,cost,c = 'blue',label='Scatter plot')
	plt.xlabel('iterations')
	plt.ylabel('Cost')	
	plt.show()
	return [w1,b]	

def myplot(head_size,brain_weight,w1,b):
	y = w1 * head_size + b
	lab = 'Error ' + str(mean_square_error(head_size,brain_weight,w1,b))
	plt.plot(head_size,y,color='red',label = lab)
	plt.scatter(head_size,brain_weight,c = 'blue',label='Scatter plot')
	plt.xlabel('Head Size in cm^3')
	plt.ylabel('Brain weight in grams')
	plt.legend()
	plt.show()

def sigmoid(x):
	return 1/(1 + np.exp(-x))

def normalize(vector):
	mean = np.mean(vector)
	min_v = np.min(vector)
	max_v = np.max(vector)
	range_v = max_v - min_v
	arr = np.array([(x-mean)/range_v for x in vector])
	return arr

def main():
	data = pd.read_csv('headbrain.csv')
	print data.shape
	print data.head()
	head_size = data['Head Size(cm^3)'].values
	mean_head = np.mean(head_size)
	range_v = np.max(head_size) - np.min(head_size)
	head_size = normalize(head_size)
	brain_weight = data['Brain Weight(grams)'].values
	mean_brain = np.mean(brain_weight)
	range_b = np.max(brain_weight) - np.min(brain_weight)
	brain_weight = normalize(brain_weight)
	print np.min(brain_weight)
	print np.max(brain_weight)
	w1 = 0.0
	b = 0
	#myplot(head_size,brain_weight,w1,b)
	#g = gradient_w1(head_size,brain_weight,w1,b)
	#print  -0.0000001 * g
	no_of_iterations = 10000
	learning_rate_w1 = 0.01
	learning_rate_b = 0.01
	[fit_w1, fit_b] = learn_model(w1,b,head_size,brain_weight,no_of_iterations,learning_rate_w1,learning_rate_b)
	#fit_w1 = 0.7543
	#fit_b = -5.24147064242e-17
	print 'Model'
	print fit_w1
	print fit_b
	print '--'
	myplot(head_size,brain_weight,fit_w1,fit_b)
	new_data = 3750
	
	new_data = (new_data - mean_head)/(range_v)
	print 'new data'
	print new_data
	print 'mean_brain', mean_brain
	print 'prediction'
	print (prediction(fit_w1,fit_b,new_data) + mean_brain)


if __name__ == '__main__':
	main()

# Tried with learning rate of 0.01. Was getting nan and exponential gradients. The gradient was in 
# lakhs and thousands. So, learning rate of 0.01 * gradient changed the learning rate to 0.00000001 
# to account for this. Got a good model. However, b was not learning at a fast rate due to slow learning 
# rate. So, used a different learning rate for b. 
# Got the perfect fit line with slope w1 = 0.263 and bias b = 325.337

# Might not be the correct approach to change the learning rate. I think it has something to be done
# with Normalizing the data.
# Ask the Professor and clear the doubt
# To do :
# how to denormalize data
#
# Andrew NG course - normalize data according to this (xi - mean)/range
# Plot Cost vs iterations to see the Gradient descent if it is working 
#
