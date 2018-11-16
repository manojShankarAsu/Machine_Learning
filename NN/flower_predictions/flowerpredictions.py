from matplotlib import pyplot as plt
import numpy as np


def NN(m1, m2, w1, w2,b):
	z = m1 * w1 + m2 * w2 + b
	return sigmoid(z)

def sigmoid(x):
	return 1/(1 + np.exp(-x))

def sigmoid_p(x):
	return sigmoid(x) * (1 - sigmoid(x))

def predict(length, wid,w1,w2,b):
	label = NN(length,wid,w1,w2,b)
	print label

def cost(prediction , target):
	return (prediction-target) ** 2

def slope(prediction,target):
	return 2 * (prediction - target)

def initial_basics():
	w1 = np.random.randn()
	w2 = np.random.randn()
	b = np.random.randn()
	print 'Parameters'
	print w1
	print w2
	print b
	length = [3,2,4,3]
	width = [1.5,1,1.5,1]
	print 'Predictions'
	for x in xrange(0,4):
		predict(length[x] , width[x] , w1, w2, b)


def mse(data,w1,w2,b):
	total_cost = 0
	for point in data:
		z = point[0] * w1 + point[1] * w2 + b
		prediction = sigmoid(z)
		target = point[2]
		total_cost += np.square(prediction - target)
	return total_cost / len(data)


def neural_network(data,w1,w2,b,learning_rate=0.2):
	#     o     flower type (output/prediction) = w1 * length + w2 * width + b
	#    /  \    w1, w2, b
	#   o    o   length, width (inputs)

	# scatter plot of data
	#print 'In neural_network'
	# for i in range(0,len(data)):
	# 	point = data[i]
	# 	color = 'r'
	# 	if point[2] == 0:
	# 		color = 'b'		
	# 	plt.scatter(point[0],point[1],c=color)
	# plt.show()

	# training 
	for i in range(100000):
		dcost_w1 = 0
		dcost_w2 = 0
		dcost_b = 0
		for rand_point in data:
			z = rand_point[0] * w1 + rand_point[1] * w2 + b
			prediction = sigmoid(z)
			target = rand_point[2]
			#cost = np.square(prediction - target) # (prediction - target) ^ 2
			#print rand_point, cost
			
			# partial derivatives of cost 
			dcost_pred = 2 * (prediction - target)
			dpred_z = sigmoid_p(z)
			dz_w1 = rand_point[0]
			dz_w2 = rand_point[1]
			dz_b=1
			dcost_w1 += dcost_pred * dpred_z * dz_w1
			dcost_w2 += dcost_pred * dpred_z * dz_w2
			dcost_b += dcost_pred * dpred_z * dz_b

		dcost_w1 /= len(data)
		dcost_w2 /= len(data)
		dcost_b /= len(data)

		w1 = w1 - learning_rate * dcost_w1
		w2 = w2 - learning_rate * dcost_w2
		b = b - learning_rate * dcost_b
		if i % 10000 == 0:
			print(mse(data,w1,w2,b))
	print(mse(data,w1,w2,b))
	return [w1,w2,b]


def prediction(point, w1, w2, b):
	z = point[0] * w1 + point[1] * w2 + b
	prediction = sigmoid(z)
	return prediction


if __name__ == '__main__':
	#initial_basics()
	data = [[3,1.5,1],[2,1,0],[4,1.5,1],[3,1,0],[3.5,0.5,1],[2,0.5,0],[5.5,1,1],[1,1,0]]
	# [length, width, 0/1] 0 - blue, 1 - red 
	new_flower = [4.5,1]
	w1 = 0
	w2 = 0
	b = 0
	[opw1,opw2, opb] = neural_network(data,w1,w2,b)
	print 'Optimal values'
	print opw1
	print opw2
	print opb

	

	# 
	print 'Test on train data'
	for point in data:
		ans = prediction(point,opw1,opw2,b)
		print 'prediction'
		print ans
		print 'Actual = '
		print point[2]
		print '---------------'


	z = new_flower[0] * opw1 + new_flower[1] * opw2 + opb
	prediction = sigmoid(z)
	print 'prediction'
	print prediction



