''' 
	This code is but a slight adaptation of Yumi's awesome code, from:
	https://fairyonice.github.io/Measure-the-uncertainty-in-deep-learning-models-using-dropout.html

	The paper of Yarin Gal and Zoubin Ghahramani explains why this works:
	"Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning"
	https://arxiv.org/pdf/1506.02142.pdf
 '''

import numpy as np
np.random.seed(42)

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
	
import matplotlib.pyplot as plt
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense,Dropout
from keras import optimizers

class KerasDropoutPrediction(object):
    def __init__(self,model):
        self.f = K.function(
                [model.layers[0].input, 
                 K.learning_phase()],
                [model.layers[-1].output])
    def predict(self,x, n_iter=100):
        result = []
        for _ in range(n_iter):
            result.append(self.f([x , 1]))
        result = np.array(result).reshape(n_iter,len(x)).T
        return result


def read_external_dataset(path):
	dataset = np.loadtxt(path)
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
	n_test = 1000
	X_train = np.random.uniform( -4.0, 4.0, n_train )
	X_train = np.sort(X_train)
	y_train = np.power(X_train, 3.0) + np.random.normal(0.0, 3.0, len(X_train))
	X_test = np.random.uniform( -6.0, 6.0, n_test )
	X_test = np.sort( X_test )
	y_test = np.power( X_test, 3.0 )

	X_train = X_train.reshape((len(X_train),1))
	X_test = X_test.reshape((len(X_test),1))

	return X_train, y_train, X_test, y_test



X_train, y_train, X_test, y_test = make_toy_dataset()
n_features = X_train.shape[1]


def define_model(rate_dropout=0.5,
                 activation="relu",
                 dropout_all_hlayers=True):

    tinput = Input(shape=(n_features,), name="ts_input")
    Nhlayer = 4
    network = tinput
    for i in range(Nhlayer):
        network = Dense(500, activation=activation)(network)
        if dropout_all_hlayers or i == (Nhlayer - 1):
            network = Dropout(rate_dropout)(network)

    out = Dense(1, activation="linear")(network)

    model = Model(inputs=[tinput], outputs=out)
    model.compile(loss="mean_squared_error",
                  optimizer=optimizers.Adam(lr=0.01))
    return model

model1 = define_model(rate_dropout=0.33)

hist = model1.fit(X_train,
	y_train,
	batch_size=len(X_train),
	verbose=True,
	epochs=1000)


kdp = KerasDropoutPrediction(model1)
y_pred_do = kdp.predict(X_test, n_iter=10)

'''
# Yumi's plots
for key in hist.history.keys():
    plt.plot(hist.history[key],label=key)
plt.title("loss={:5.4f}".format(hist.history["loss"][-1]))
plt.legend()
plt.yscale('log')
plt.show()


y_pred_do_mean = y_pred_do.mean(axis=1)

y_pred = model1.predict(X_test) 

plt.figure(figsize=(5,5))
plt.scatter(y_pred_do_mean , y_pred, alpha=0.1)
plt.xlabel("The average of dropout predictions")
plt.ylabel("The prediction without dropout from Keras")
plt.show()


def vertical_line_trainin_range(ax):
    minx, maxx = np.min(X_train), np.max(X_train)
    ax.axvline(maxx,c="red",ls="--")
    ax.axvline(minx,c="red",ls="--",
               label="The range of the X_train")

def plot_y_pred_do(ax,y_pred_do, 
                   fontsize=20,
                   alpha=0.05,
                   title="The 100 y values predicted with dropout"):
    for iiter in range(y_pred_do.shape[1]):
        ax.plot(X_test,y_pred_do[:,iiter],label="y_pred (Dropout)",alpha=alpha)

    vertical_line_trainin_range(ax)
    ax.set_title(title,fontsize= fontsize)

    
       
fig = plt.figure(figsize=(10,5))
fig.subplots_adjust( hspace = 0.13 , wspace = 0.05)

ax = fig.add_subplot(1,2,1)
ax.plot(X_test,y_pred,  
        color="yellow",
        label = "y_pred (from Keras)")
ax.scatter(X_test,y_pred_do_mean, 
           s=50,alpha=0.05,
           color="magenta",
           label = "y_pred (Dropout average)")
ax.scatter(X_train, y_train, label = "y_train")

ax.set_xlabel("x")
vertical_line_trainin_range(ax)
ax.legend()

ax = fig.add_subplot(1,2,2)
plot_y_pred_do(ax,y_pred_do)
'''

fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(1,1,1)
ax.scatter(np.squeeze(X_train), y_train, color='purple')
ax.plot(np.arange(-6.0,7.0), np.power(np.arange(-6.0,7.0),3.0), color='red')
	
for iiter in range(y_pred_do.shape[1]):
        ax.plot(X_test,y_pred_do[:,iiter],label="y_pred (Dropout)",alpha=0.05, color='blue')

plt.ylim(top=np.max(y_test))  
plt.ylim(bottom=np.min(y_test))

fig.tight_layout()

plt.show()
