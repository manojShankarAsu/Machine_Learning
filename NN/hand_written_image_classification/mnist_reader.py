import os
import gzip
import numpy as np


def load_mnist(path, kind='train'):
  
  labels_path = os.path.join(path,'%s-labels-idx1-ubyte.gz'% kind)
  images_path = os.path.join(path,'%s-images-idx3-ubyte.gz'% kind)

  with gzip.open(labels_path, 'rb') as lbpath:
    labels = np.frombuffer(lbpath.read(), dtype=np.uint8,offset=8)
  with gzip.open(images_path, 'rb') as imgpath:
    images = np.frombuffer(imgpath.read(), dtype=np.uint8,offset=16).reshape(len(labels), 784)
  return images, labels


def load_mnist_wrapper(path,kind='train'):
	(images,labels) = load_mnist(path,kind)
	n = images.shape[0]
	train_image = [np.reshape(images[i,:] , (784,1)) for i in xrange(n)]
	training_data = zip(train_image,train_image)
	return training_data

