from __future__ import print_function
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import os
warnings.filterwarnings('ignore')

from keras.models import Model
from keras.layers import Dense, Input
from keras.datasets import mnist
from keras.datasets import fashion_mnist
from keras.regularizers import l1
from keras.optimizers import Adam


def main():
	i = 1
	#(x_train, y_train), (x_test, y_test) = mnist.load_data()
	(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
	#print(x_train.shape)
	x_train = x_train.astype('float32') / 255.0
	x_test = x_test.astype('float32') / 255.0
	x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
	x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
	print(x_train.shape)
	print(x_test.shape)
	#print(x_train[])
	noise_factor_train = 0.4
	noise_factor_test = 0.4
	x_train_noisy = x_train + noise_factor_train * np.random.normal(size=x_train.shape)
	x_test_noisy = x_test + noise_factor_test * np.random.normal(size=x_test.shape)

	x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
	x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)
	n = 5
	curr_d = os.getcwd()
	curr_d = os.path.join(curr_d , 'images')
	#plt.figure(figsize=(10, 4.5))
	orig_str= 'original_{0}'
	noise_str= 'noise_{0}'
	dec_str = 'deconstructed_{0}'
	# for i in range(n):
	#     # plot original image
	#     #ax = plt.subplot(2, n, i + 1)
	# 	# plt.imshow()
	# 	# plt.gray()
	# 	# plt.show()
	#     plt.matshow(x_test[i].reshape(28,28))
	#     plt.savefig(os.path.join(curr_d,orig_str.format(i)))
	#     # ax.get_xaxis().set_visible(False)
	#     # ax.get_yaxis().set_visible(False)
	#     # if i == n/2:
	#     #     ax.set_title('Original Images')

	#     # plot noisy image 
	#     # ax = plt.subplot(2, n, i + 1 + n)
	# 	# plt.imshow(x_test_noisy[i].reshape(28, 28))
	# 	# plt.gray()
	# 	# plt.show()
	#     plt.matshow(x_test_noisy[i].reshape(28, 28))
	#     plt.savefig(os.path.join(curr_d,noise_str.format(i)))
	#     # ax.get_xaxis().set_visible(False)
	#     # ax.get_yaxis().set_visible(False)
	#     # if i == n/2:
	#     #     ax.set_title('Noisy Input')


	input_size = 784
	hidden_size = 500

	input_img = Input(shape=(input_size,))
	hidden_1 = Dense(hidden_size, activation='relu')(input_img)
	output_img = Dense(input_size, activation='sigmoid')(hidden_1)

	autoencoder = Model(input_img, output_img)
	autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
	autoencoder.fit(x_train_noisy, x_train, epochs=10)

	n = 30
	plt.figure(figsize=(10, 7))

	images = autoencoder.predict(x_test_noisy)
	for i in range(n):
	    # plot original image
	    plt.matshow(x_test[i].reshape(28,28))
	    plt.savefig(os.path.join(curr_d,orig_str.format(i)))
	 

	    # plot noisy image 
	    plt.matshow(x_test_noisy[i].reshape(28, 28))
	    plt.savefig(os.path.join(curr_d,noise_str.format(i)))

	    # plot decoded image 
	    plt.matshow(images[i].reshape(28, 28))
	    plt.savefig(os.path.join(curr_d,dec_str.format(i)))

if __name__ == '__main__':
	main()