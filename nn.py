import numpy as np
import pandas as pd
import tensorflow as tf
from featureSelection import cleanData, principalComponent
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import explained_variance_score
from sklearn.utils import resample
#only 100 samples so there is no need to batch data.

n_input = 36
n_hidden1 = 500
n_hidden2 = 800
n_hidden3 = 50
n_hidden4 = 30
n_output = 10
lr = 0.05
n_epoch = 300
BATCH_SIZE = 100

 

filename = 'integratedDataset_modified.csv'
dataset, colName = cleanData(filename)
dataResampled = resample(dataset, n_samples = 1000)

X = dataResampled.iloc[:,:36] # adjust according to the current number of targets.
y = dataResampled.iloc[:,36:]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state = 42) # splitting dataset into training and testing.

scaler = StandardScaler().fit(X_train) # standardizing features.
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)# apply the same mean and variance from traning set to testing set.
y_train = np.asarray(y_train).reshape(-1,n_output)
y_test = np.asarray(y_test).reshape(-1,n_output)

num_batches = int(len(X_train)/BATCH_SIZE*1.0) # making batches 
print num_batches


# create placesholders for data to feed in
X = tf.placeholder(tf.float32, shape = (None, n_input), name = 'X')
y = tf.placeholder(tf.float32, shape = (None, n_output), name = 'y')
batch_size = tf.placeholder(tf.int64)

data = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size).repeat()
iter = data.make_initializable_iterator()
features, labels = iter.get_next()

# define between layer calcualtion
def neruon_layer(X, n_neurons, name, activition = None):
	with tf.name_scope(name):
		n_input = int(X.get_shape()[1])
		stddev = 2/np.sqrt(n_input)
		init = tf.truncated_normal((n_input, n_neurons), stddev = stddev)
		W = tf.Variable(init, name = 'kernel')
		b = tf.Variable(tf.zeros([n_neurons]), name = 'bias')
		Z = tf.matmul(X,W) + b
		if activition is not None:
			return activition(Z)
		else:
			return Z

# construct the network bulk
with tf.name_scope('dnn'):
	hidden1 = neruon_layer(features, n_hidden1, name = 'hidden1', activition = tf.nn.relu)
	dropout = tf.nn.dropout(hidden1, keep_prob = 0.5)
	hidden2 = neruon_layer(dropout, n_hidden2, name = 'hidden2', activition = tf.nn.relu)
	dropout2 = tf.nn.dropout(hidden2, keep_prob = 0.5)
	hidden3 = neruon_layer(dropout2, n_hidden3, name = 'hidden3', activition = tf.nn.relu)
	dropout3 = tf.nn.dropout(hidden3, keep_prob = 0.5)
	hidden4 = neruon_layer(dropout3, n_hidden4, name = 'hidden4', activition = tf.nn.relu)
	dropout4 = tf.nn.dropout(hidden4, keep_prob = 0.5)
	output = neruon_layer(dropout, n_output, name = 'output', activition = None)

# define loss function
with tf.name_scope('loss'):
	mse = tf.losses.mean_squared_error(labels = labels, predictions = output)

# training process
with tf.name_scope('train'):
	optimizer = tf.train.AdamOptimizer()
	training_op = optimizer.minimize(mse)

# evaluation
with tf.name_scope('eval'):
	loss_val = tf.losses.mean_squared_error(labels = labels, predictions = output)
	acc = tf.cast(loss_val, tf.float32)

initializer = tf.global_variables_initializer()
saver = tf.train.Saver()

# execution
with tf.Session() as sess:
	sess.run(initializer)
	
	for epoch in range(n_epoch):
		sess.run(iter.initializer, feed_dict = {X: X_train, y: y_train, batch_size: BATCH_SIZE})
		train_error = []
		for i in range(num_batches):
			sess.run(training_op)
			train_error.append(mse.eval())
			
		sess.run(iter.initializer, feed_dict = {X: X_test, y: y_test, batch_size: len(y_test)})
		acc_test = acc.eval()
		print(epoch + 1, "Train loss", np.mean(train_error), "Test Loss: %.4f" % acc_test)
	#save_path = saver.save(sess,"./my_model_final.ckpt")


