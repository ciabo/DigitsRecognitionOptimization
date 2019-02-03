import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy import interpolate
import random
import RBF as r

# keras imports for the dataset and building our neural network
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import Callback
from keras import backend as K
from keras import optimizers
from keras.models import model_from_json


class learningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = float(K.get_value(self.model.optimizer.lr))
        print("Learning rate:", "%.4f" % lr)


def plot_history(history):
    loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' not in s]
    val_loss_list = [s for s in history.history.keys() if 'loss' in s and 'val' in s]
    acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' not in s]
    val_acc_list = [s for s in history.history.keys() if 'acc' in s and 'val' in s]

    if len(loss_list) == 0:
        print('Loss is missing in history')
        return

        # As loss always exists
    epochs = range(1, len(history.history[loss_list[0]]) + 1)

    # Loss
    plt.figure(1)
    for l in loss_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))
    for l in val_loss_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation loss (' + str(str(format(history.history[l][-1], '.5f')) + ')'))

    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy
    plt.figure(2)
    for l in acc_list:
        plt.plot(epochs, history.history[l], 'b',
                 label='Training accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')
    for l in val_acc_list:
        plt.plot(epochs, history.history[l], 'g',
                 label='Validation accuracy (' + str(format(history.history[l][-1], '.5f')) + ')')

    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def plot_rbf(x_val, y_val, rbfi):
    xnew = np.linspace(0.001, 2, 100)
    fval = rbfi(xnew)

    plt.figure(3)
    plt.scatter(x_val, y_val, c='r', marker='o')
    plt.plot(xnew, fval, label="gaussian")
    plt.title('Rbf interpolation')
    plt.ylabel('Rbf')
    plt.legend()
    plt.show()


# Load and prepare the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# building the input vector from the 28x28 pixels
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train, x_test = x_train / 255.0, x_test / 255.0

# building a linear stack of layers with the sequential model
model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

model.add(Dense(10))
model.add(Activation('softmax'))

# exporting the model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

'''
#finds first three results
startersPoints=np.array([])
starterValues=np.array([])
for i in range(0,3):
    # load json and create model in this way we will use always the same model and weights
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")


    learning_rate=random.uniform(0.001,2)
    startersPoints= np.append(startersPoints,learning_rate)

    opt = optimizers.Adam(lr=learning_rate) #default decay=0
    loaded_model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # training the model and saving metrics in history
    epochs = 5
    history = loaded_model.fit(x_train, y_train, epochs=epochs, verbose=2, callbacks=[learningRateTracker()])

    results=loaded_model.evaluate(x_test, y_test)  # return loss and precision
    print(results)

    starterValues=np.append(starterValues,results[0])
    plot_history(history)

print(startersPoints)
print(starterValues)
#THIS IS SHIT
#Non si possono avere i moltiplicatori delle funzioni di base e quindi non si puà minimizzare la bumpiness
#Credo che l'unica soluzione sia fare tutto a mano
#Anche facendo così poi c'è da minimizzare la bumpiness
# ==> siamo nella merda
rbfi=interpolate.Rbf(startersPoints,starterValues,function="gaussian")
'''
x = np.array([2,3,5,])
f = np.array([0.2,0.8,0.5])
rbfi=interpolate.Rbf(x,f,function="gaussian")
rbf=r.RBF(x,f)
rbf.interpolate()
lambd=rbf.getMultipliers()
print(lambd)
a=rbf.g(2)