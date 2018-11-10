import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of one feature.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
  
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(buffer_size=10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def run():
	tf.logging.set_verbosity(tf.logging.ERROR)
	pd.options.display.max_rows = 10
	pd.options.display.float_format = '{:.1f}'.format


	california_housing_dataframe = pd.read_csv("https://download.mlcc.google.com/mledu-datasets/california_housing_train.csv", sep=",")
	california_housing_dataframe = california_housing_dataframe.reindex(np.random.permutation(california_housing_dataframe.index))

	california_housing_dataframe["median_house_value"] /= 1000.0
	#print california_housing_dataframe.head()
	#print california_housing_dataframe.describe()

	my_feature = california_housing_dataframe[['total_rooms']]
	feature_columns = [tf.feature_column.numeric_column('total_rooms')]

	targets = california_housing_dataframe[['median_house_value']]

	my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0000001)
	my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)

	linear_regressor = tf.estimator.LinearRegressor(feature_columns = feature_columns,optimizer = my_optimizer)
	# features = {key:np.array(value) for key,value in dict(my_feature).items()}
	# print features
	_ = linear_regressor.train(input_fn = lambda:my_input_fn(my_feature,targets),steps = 100)


	prediction_input_fn =lambda: my_input_fn(my_feature, targets, num_epochs=1, shuffle=False)
	predictions = linear_regressor.predict(input_fn=prediction_input_fn)
	predictions = np.array([item['predictions'][0] for item in predictions])
	print predictions
	mean_squared_error = metrics.mean_squared_error(predictions, targets)
	root_mean_squared_error = math.sqrt(mean_squared_error)
	print "Mean Squared Error (on training data): %0.3f" % mean_squared_error
	print "Root Mean Squared Error (on training data): %0.3f" % root_mean_squared_error

	min_house_value = california_housing_dataframe["median_house_value"].min()
	max_house_value = california_housing_dataframe["median_house_value"].max()
	min_max_difference = max_house_value - min_house_value

	print "Min. Median House Value: %0.3f" % min_house_value
	print "Max. Median House Value: %0.3f" % max_house_value
	print "Difference between Min. and Max.: %0.3f" % min_max_difference
	print "Root Mean Squared Error: %0.3f" % root_mean_squared_error

if __name__ == '__main__':
	run()