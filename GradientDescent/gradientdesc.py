from numpy import *


def compute_error_for_given_points(b,m,points):
	error = 0
	for i in range(0,len(points)):
		x = points[i,0]
		y = points[i,1]
		curr_error = y - (m * x) - b
		error = error + (curr_error * curr_error)
	return error / float(len(points))

def step_gradient(b_current,m_current,points, learning_rate):
	b_gradient = 0
	m_gradient = 0
	N = float(len(points))
	for i in range(0,len(points)):
		x = points[i,0]
		y = points[i,1]
		m_gradient = m_gradient + ((y - m_current * x - b_current) * x) 
		b_gradient = b_gradient + (y - m_current * x - b_current)
	m_gradient = (-2 * m_gradient) / N # gives the direction of optimum m 
	b_gradient = (-2 * b_gradient) / N # gives the direction of optimum b 

	new_b = b_current - (learning_rate * b_gradient)  # negative gradient gives the direction of steepest descent
	new_m = m_current - (learning_rate * m_gradient)
	return [new_b , new_m]


def gradient_descent_runner(points, learning_rate, b_initial, m_initial, num_iterations):
	b = b_initial
	m = m_initial

	print 'Errors:'
	for i in range(num_iterations):
		b,m = step_gradient(b,m,array(points),learning_rate)
		print compute_error_for_given_points(b,m,points)
	return [b,m]

def run():
	points = genfromtxt('data.csv',delimiter=',')
	# hyperparamters
	learning_rate = 0.0002
	# y= mx + b
	b_initial = 0
	m_initial = 0
	num_iterations = 1000
	[b,m] = gradient_descent_runner(points,learning_rate,b_initial,m_initial,num_iterations)
	print 'Best fit line'
	print 'm='+str(m)
	print 'b='+str(b)



if __name__ == '__main__':
	run()
