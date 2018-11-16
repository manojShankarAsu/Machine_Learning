import random
import numpy as np
from mnist_loader import load_data_wrapper
from load_mnist import mnist
import sys

def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z)) 

def sigmoid_prime(z):
	return sigmoid(z) * (1 - sigmoid(z))

class NeuralNetwork(object):

	def __init__(self,sizes):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.zeros((y,1)) for y in sizes[1:]] 
		# between two layers , the no of bias = no of neurons in the next layer
		self.weights = [np.zeros((y,x)) for x,y in zip(sizes[:-1],sizes[1:])]
		# ignore the last layer. from layer 1 to n-1 , 
		# weight matrix dimension = no of next neurons (rows) * no of current layer neurons (columns)
		# size[:-1] ignores last layer. sizes[1:] starts from second layer.

	def feedforward(self, a):
		for b,w in zip(self.biases,self.weights):
			a = sigmoid(np.dot(w,a) + b)
		return a

	def BatchGD(self,training_data,epochs,learning_rate,test_data=None):
		count =1 
		if test_data:
			n_test = len(test_data)
		n = len(training_data)
		for j in xrange(epochs):
			random.shuffle(training_data)
			self.update_batch(training_data,learning_rate,count)
			if test_data:
				print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data,count), n_test)
			else:
				print "Epoch {0} complete".format(j)
			count += 1

	# def SGD(self,training_data,epochs,mini_batch_size,learning_rate,test_data=None):
	# 	if test_data:
	# 		n_test = len(test_data)
	# 	n = len(training_data)
	# 	for j in xrange(epochs):
	# 		random.shuffle(training_data)
	# 		mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0,n,mini_batch_size)]
	# 		for mini_batch in mini_batches:
	# 			self.update_batch(mini_batch,learning_rate)
	# 		if test_data:
	# 			print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
	# 		else:
	# 			print "Epoch {0} complete".format(j)

	def update_batch(self,mini_batch,learning_rate,count):
		print 'Iteration {0}'.format(count)
		total_gradient_b = [np.zeros(b.shape) for b in self.biases]
		total_gradient_w = [np.zeros(w.shape) for w in self.weights]
		for x, y in mini_batch:
			delta_b, delta_w = self.backprop(x,y,count)
			total_gradient_b = [ t_g_b + db for t_g_b,db in zip(total_gradient_b,delta_b)]
			total_gradient_w = [ t_g_w + dw for t_g_w, dw in zip(total_gradient_w,delta_w)]
		print 'Total Gradients'
		print total_gradient_w[0]
		print total_gradient_b[0]
		self.weights = [w-(learning_rate/len(mini_batch))*gw for w, gw in zip(self.weights,total_gradient_w)]
		self.biases = [b- (learning_rate/len(mini_batch))*gb for b, gb in zip(self.biases,total_gradient_b)]

	def backprop(self,x,y,count):
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

		print 'Last layer:'
		print activations[-1]

		# backward propogation
		del_l = self.derivative_cost_wrt_aL(activations[-1],y) * sigmoid_prime(zs[-1])
		print 'del_l {0}'.format(del_l)
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

	def evaluate(self,test_data,count):
		
		test_results = [(self.feedforward(x),y) for (x,y) in test_data]
		#return sum(int(x == y) for (x,y) in test_results)		
		ret = []
		for (x,y) in test_results:			
			if y == 0:
				if x < 0.5:
					ret.append(1)
			else:
				if x >= 0.5:
					ret.append(1)
		return np.sum(ret)


def process_data(data,labels,noSamples):
	my_data = [np.reshape(data[:,x] , (784,1)) for x in xrange(noSamples)]
	my_label = [np.array(x) for x in labels[0]]
	my_dataset = zip(my_data,my_label)
	return my_dataset

def main():
	dimension_string = sys.argv[1]
	dimensions = map(int,dimension_string.strip('[]').split(','))
	net = NeuralNetwork(dimensions) 
	#net.SGD(training_data,30,10,3.0,test_data=test_data)

	digit_range = [1]
	noTrSamples = 1
	noTsSamples = 1
	train_data, train_label, test_data, test_label = mnist(noTrSamples=noTrSamples,noTsSamples=noTsSamples, \
    													digit_range = digit_range,noTrPerClass=1,noTsPerClass=1)

	#convert to binary labels
	train_label[train_label==digit_range[0]] = 1
	#train_label[train_label==digit_range[1]] = 1
	test_label[test_label==digit_range[0]] = 1
	#test_label[test_label==digit_range[1]] = 1

	my_train_data =  process_data(train_data,train_label,noTrSamples)
	my_test_data = process_data(test_data,test_label,noTsSamples)
	net.BatchGD(my_train_data,30,0.1,test_data = my_test_data)


if __name__ == '__main__':
	main()