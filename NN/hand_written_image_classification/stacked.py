from __future__ import print_function
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

from keras.models import Model,Sequential
from keras.layers import Dense, Input
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.regularizers import l1
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


def get_onehot_vector(y_train):
	# encode class values as integers
	encoder = LabelEncoder()
	encoder.fit(y_train)
	encoded_Y = encoder.transform(y_train)
	# convert integers to dummy variables (i.e. one hot encoded)
	one_hot_vector = np_utils.to_categorical(encoded_Y)
	return one_hot_vector

def baseline_model():
	classifier = Sequential()
	classifier.add(Dense(10, input_dim=100, activation='softmax'))
	# Compile model
	classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return classifier

def main():
	i = 1
	#(x_train, y_train), (x_test, y_test) = mnist.load_data()
	(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


	#print(x_train.shape)
	print('y_train')
	print(y_train[0])	
	x_train = x_train.astype('float32') / 255.0
	x_test = x_test.astype('float32') / 255.0

	# x_train,x_valid,train_ground,valid_ground = train_test_split(x_train,
 #                                                             x_train,
 #                                                             test_size=0.167, 
 #                                                             random_state=13)

	x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
	x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
	# x_valid = x_valid.reshape((len(x_valid), np.prod(x_valid.shape[1:])))
	print(x_train.shape)
	# print(x_valid.shape)

	


	# print(x_train[])
	my_epochs = 10
	
	curr_d = os.getcwd()
	curr_d = os.path.join(curr_d , 'images')
	#plt.figure(figsize=(10, 4.5))
	orig_str= 'original_{0}'
	noise_str= 'noise_{0}'
	dec_str = 'deconstructed_{0}'

	input_size = 784
	hidden_size = 500
	loss_function = 'mean_squared_error'

	input_img = Input(shape=(input_size,))
	hidden_1 = Dense(hidden_size, activation='sigmoid')(input_img)
	output_img = Dense(input_size, activation='sigmoid')(hidden_1)

	autoencoder_1 = Model(input_img, output_img)
	autoencoder_1.compile(optimizer='adam', loss=loss_function)

	autoencoder_train_1 = autoencoder_1.fit(x_train, x_train, epochs=my_epochs)

	# loss = autoencoder_train.history['loss']
	# val_loss = autoencoder_train.history['val_loss']
	# epochs = range(10)
	# plt.figure()
	# plt.plot(epochs, loss, 'bo', label='Training loss')
	# plt.plot(epochs, val_loss, 'b', label='Validation loss')
	# plt.title('Training and validation loss')
	# plt.legend()
	# plt.show()

	# n = 1
	# plt.figure(figsize=(10, 7))

	# images = autoencoder_1.predict(x_train)
	# for i in range(n):
	#     # plot original image
	#     # plt.matshow(x_test[i].reshape(28,28))
	#     # #plt.savefig(os.path.join(curr_d,orig_str.format(i)))
	#     # plt.show()
	 

	#     # # plot noisy image 
	#     # plt.matshow(x_test_noisy[i].reshape(28, 28))
	#     # #plt.savefig(os.path.join(curr_d,noise_str.format(i)))
	#     # plt.show()

	#     # plot decoded image 
	#     plt.matshow(images[i].reshape(28, 28))
	#     #plt.savefig(os.path.join(curr_d,dec_str.format(i)))
	#     plt.show()
	

	layer_500 = Model(inputs=autoencoder_1.input,outputs=autoencoder_1.layers[1].output)
	features_500 = layer_500.predict(x_train)


	# autoencoder 2
	input_img_2 = Input(shape=(500,))
	hidden_2 = Dense(200, activation='sigmoid')(input_img_2)
	output_img_2 = Dense(500, activation='sigmoid')(hidden_2)

	autoencoder_2 = Model(input_img_2, output_img_2)
	autoencoder_2.compile(optimizer='adam', loss=loss_function)

	autoencoder_train_2 = autoencoder_2.fit(features_500, features_500, epochs=my_epochs)

	layer_200 = Model(inputs=autoencoder_2.input,outputs=autoencoder_2.layers[1].output)
	features_200 = layer_200.predict(features_500)
	print('Features 200 ')
	print(features_200.shape)

	# autoencoder 3
	# autoencoder 2
	input_size = 200
	input_img_3 = Input(shape=(input_size,))
	hidden_3 = Dense(100, activation='sigmoid')(input_img_3)
	output_img_3 = Dense(input_size, activation='sigmoid')(hidden_3)

	autoencoder_3 = Model(input_img_3, output_img_3)
	autoencoder_3.compile(optimizer='adam', loss=loss_function)

	autoencoder_train_3 = autoencoder_3.fit(features_200, features_200, epochs=my_epochs)

	layer_100 = Model(inputs=autoencoder_3.input,outputs=autoencoder_3.layers[1].output)
	features_100 = layer_100.predict(features_200)
	print('Features 100 ')
	print(features_100.shape)

	one_hot_vector = get_onehot_vector(y_train)
	test_one_hot = get_onehot_vector(y_test)
	

	
	classifier = Sequential()
	classifier.add(Dense(32, activation='relu', input_dim=100))
	classifier.add(Dense(10, activation='softmax'))
	classifier.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
	classifier.fit(features_100,one_hot_vector,epochs=10)

	test_500=layer_500.predict(x_test)
	test_200=layer_200.predict(test_500)
	test_100=layer_100.predict(test_200)
	score = classifier.evaluate(test_100, test_one_hot)
	print('Score:')
	print(score)


	# n = 30
	# plt.figure(figsize=(10, 7))

	# images = autoencoder.predict(x_test_noisy)
	# for i in range(n):
	#     # plot original image
	#     plt.matshow(x_test[i].reshape(28,28))
	#     plt.savefig(os.path.join(curr_d,orig_str.format(i)))
	 

	#     # plot noisy image 
	#     plt.matshow(x_test_noisy[i].reshape(28, 28))
	#     plt.savefig(os.path.join(curr_d,noise_str.format(i)))

	#     # plot decoded image 
	#     plt.matshow(images[i].reshape(28, 28))
	#     plt.savefig(os.path.join(curr_d,dec_str.format(i)))



if __name__ == '__main__':
	main()