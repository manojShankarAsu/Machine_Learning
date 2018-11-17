import random
import numpy as np
from load_mnist import mnist
from mnist_loader import load_data_wrapper
import sys
import matplotlib.pyplot as plt

def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z)) 

def sigmoid_prime(z):
	return sigmoid(z) * (1 - sigmoid(z))

def square(z):
	return z ** 2

class NeuralNetwork(object):

	def __init__(self,sizes):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y,1) for y in sizes[1:]] 
		#self.biases = [np.zeros((y,1)) for y in sizes[1:]] 
		# between two layers , the no of bias = no of neurons in the next layer
		self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
		#self.weights = [np.zeros((y,x)) for x,y in zip(sizes[:-1],sizes[1:])]
		# ignore the last layer. from layer 1 to n-1 , 
		# weight matrix dimension = no of next neurons (rows) * no of current layer neurons (columns)
		# size[:-1] ignores last layer. sizes[1:] starts from second layer.

	def feedforward(self, a):
		for b,w in zip(self.biases,self.weights):
			a = sigmoid(np.dot(w,a) + b)
		return a

	def cost(self,training_data):
		costt = 0
		for x,y in training_data:
			prediction = self.feedforward(x)
			diff = prediction - y
			diff = square(diff)
			costt += np.sum(diff)
		return costt / len(training_data)



	def BatchGD(self,training_data,epochs,learning_rate,test_data=None):
		if test_data:
			n_test = len(test_data)
		n = len(training_data)
		iterations = []
		cost_iter = []
		for j in xrange(epochs):
			#random.shuffle(training_data)
			self.update_batch(training_data,learning_rate)
			if test_data:
				print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
			else:
				print "Epoch {0} complete".format(j)
				iterations.append(j)
				tmp = self.cost(training_data)
				cost_iter.append(tmp)
				print tmp
		plt.scatter(iterations,cost_iter,c = 'blue',label='Scatter plot')
		plt.xlabel('iterations')
		plt.ylabel('Cost')
		plt.show()



	def SGD(self,training_data,epochs,mini_batch_size,learning_rate,test_data=None):
		if test_data:
			n_test = len(test_data)
		n = len(training_data)
		iterations = []
		cost_iter = []
		for j in xrange(epochs):
			random.shuffle(training_data)
			mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0,n,mini_batch_size)]
			for mini_batch in mini_batches:
				self.update_batch(mini_batch,learning_rate)
			if test_data:
				print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
			else:
				print "Epoch {0} complete".format(j)
				iterations.append(j)
				tmp = self.cost(training_data)
				cost_iter.append(tmp)
				print tmp
		plt.scatter(iterations,cost_iter,c = 'blue',label='Scatter plot')
		plt.xlabel('iterations')
		plt.ylabel('Cost')
		plt.show()

	def update_batch(self,mini_batch,learning_rate):
		total_gradient_b = [np.zeros(b.shape) for b in self.biases]
		total_gradient_w = [np.zeros(w.shape) for w in self.weights]
		for x, y in mini_batch:
			delta_b, delta_w = self.backprop(x,y)
			total_gradient_b = [ t_g_b + db for t_g_b,db in zip(total_gradient_b,delta_b)]
			total_gradient_w = [ t_g_w + dw for t_g_w, dw in zip(total_gradient_w,delta_w)]
		self.weights = [w-(learning_rate/len(mini_batch))*gw for w, gw in zip(self.weights,total_gradient_w)]
		self.biases = [b- (learning_rate/len(mini_batch))*gb for b, gb in zip(self.biases,total_gradient_b)]

	def backprop(self,x,y):
		gradient_b = [np.zeros(b.shape) for b in self.biases]
		gradient_w = [np.zeros(w.shape) for w in self.weights]

		activation = x
		activations = [x]
		zs = []

		# forward propogation
		for b,w in zip(self.biases,self.weights):
			z = np.dot(w,activation) + b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)

		# backward propogation
		del_l = self.derivative_cost_wrt_aL(activations[-1],y) * sigmoid_prime(zs[-1])
		gradient_b[-1] = del_l
		gradient_w[-1] = np.dot(del_l,np.array(activations[-2]).transpose())

		for l in xrange(2,self.num_layers):
			z_l = zs[-l]
			del_l = np.dot(np.array(self.weights[-l+1]).T , del_l) * sigmoid_prime(z_l)
			gradient_b[-l]  = del_l
			gradient_w[-l] = np.dot(del_l, np.array(activations[-l-1]).T)

		return (gradient_b , gradient_w)



	def derivative_cost_wrt_aL(self,last_layer_activation,y):
		return (last_layer_activation - y)

	def evaluate(self,test_data):
		test_results = [(np.argmax(self.feedforward(x)),y) for (x,y) in test_data]
		return sum(int(x == y) for (x,y) in test_results)



def vectorized_result(j):

	try:
		e = np.zeros((10, 1))
		e[int(j)] = 1.0
		return e
	except:
		print 'Error'
		print j


def process_data(train_data,train_label):
	n = train_data.shape[1]
	image = [ np.array(np.reshape(train_data[:,i] , (784,1)) ,dtype='float32')  for i in xrange(n)]
	labels = [vectorized_result(train_label[0][i]) for i in xrange(train_label.shape[1])]
	data_with_labels = zip(image,labels)
	return data_with_labels

def main():
	dimension_string = sys.argv[1]
	dimensions = map(int,dimension_string.strip('[]').split(','))
	net = NeuralNetwork(dimensions) # map(int,input.strip('[]').split(','))
	(training_data, validation, test_data) = load_data_wrapper(5000,1000,1000)
	#print training_data[0:2]
	net.SGD(training_data,30,10,0.1)
	#net.BatchGD(training_data,30,10)
	# train_data, train_label, test_data, test_label = \
	# 		mnist(noTrSamples=5000,noTsSamples=1000,\
	# 		digit_range=[0,1,2,3,4,5,6,7,8,9],\
	# 		noTrPerClass=500, noTsPerClass=100)
	# #print train_data.shape[1]
	# my_train_data = process_data(train_data,train_label)
	# my_test_images = [np.reshape(test_data[:,i] , (784,1)) for i in xrange(test_data.shape[1])]
	# my_test_data = zip(my_test_images,test_label[0])
	# print 'PROFESSOR'
	#print my_train_data[0:2]
	#net.BatchGD(my_train_data,30,0.9)
	#net.SGD(my_train_data,30,10,3)



if __name__ == '__main__':
	main()