''' 
	Attempt to reproduce the Toy Experiment of
	Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles
	http://bit.ly/2C9Z8St
 '''

import numpy as np
np.random.seed(42)

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
	
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Activation, Input
from keras import losses
from keras import optimizers
from keras.constraints import maxnorm, nonneg
from keras.utils import np_utils
from keras import backend as K



def read_external_dataset(path):
	dataset = np.loadtxt('datasets/rw/yacht_full.dat')
	dataset = preprocessing.scale(dataset)
	n_features = dataset.shape[1] - 1
	train, test = train_test_split(dataset, test_size=0.25, random_state=42)
	X_train = train[:, 0:n_features]
	y_train = train[:, n_features]

	X_test = test[:, 0:n_features]
	y_test = test[:, n_features]

	return X_train, y_train, X_test, y_test

def make_toy_dataset():
	n_train = 20
	n_test = 30000
	X_train = np.random.uniform( -4.0, 4.0, n_train )
	X_train = np.sort(X_train)
	y_train = np.power(X_train, 3.0) + np.random.normal(0.0, 3.0, len(X_train))
	X_test = np.random.uniform( -6.0, 6.0, n_test )
	X_test = np.sort( X_test )
	y_test = np.power( X_test, 3.0 )

	X_train = X_train.reshape((len(X_train),1))
	X_test = X_test.reshape((len(X_test),1))

	return X_train, y_train, X_test, y_test


def build_mean_branch(inputs):
	x = inputs
	x = Dense(100, activation='relu')(x)
	x = Dense(1, activation='linear', name='mean_output')(x)
	return x

def built_variance_branch(inputs):
	x = inputs
	x = Dense(100, activation='relu')(x)
	x = Dense(1, activation='softplus', name='var_output')(x) 
	return x

X_train, y_train, X_test, y_test = make_toy_dataset()
n_features = X_train.shape[1]

def generateAndTrainModel(name):

	inputs = Input(shape=(n_features,))
	mean_branch = build_mean_branch(inputs)
	var_branch = built_variance_branch(inputs)
	label_mean = Input(shape=(1,))
	model = Model(inputs=[inputs, label_mean], outputs=[mean_branch, var_branch])

	# custom loss
	loss = 0.5*K.log(var_branch + 1e-6) + 0.5*K.square(label_mean - mean_branch) / (var_branch + 1e-6)

	model.add_loss( loss )
	optimizer = optimizers.Adam( lr=0.1 )
	model.compile( optimizer=optimizer, loss=None )

	model.fit( [X_train, y_train], epochs=500, batch_size=5)

	return model

def getModelPrediction(model):
	result = model.predict([X_test, y_test])
	return np.squeeze(result)


N_ensemble = 5

means = []
variances = []


# ensemble
for i in range(N_ensemble):
	model = generateAndTrainModel('model_'+str(i))
	result = getModelPrediction(model)
	means.append(result[0])
	variances.append(result[1])

meanE = np.mean(means, axis=0)
varianceE = 0

for i in range(N_ensemble):
	varianceE = varianceE + variances[i] + np.square(means[i])

varianceE = varianceE / N_ensemble - np.square(meanE)


if n_features == 1:

	import matplotlib.pyplot as plt

	fig = plt.figure(figsize=(6,4))
	ax = fig.add_subplot(111)

	ax.scatter(np.squeeze(X_train), y_train, color='purple')
	ax.plot(np.arange(-6.0,7.0), np.power(np.arange(-6.0,7.0),3.0), color='red')
	
	three_stdE = 3.0 * np.sqrt(varianceE)
	
	ax.fill_between(np.squeeze(X_test), meanE + three_stdE, meanE - three_stdE, alpha=0.2, color='blue')

	plt.ylim(top=np.max(y_test))  
	plt.ylim(bottom=np.min(y_test))

	fig.tight_layout()

	plt.show()
