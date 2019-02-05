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
        # print("Learning rate:", "%.4f" % lr)


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


def plot_rbf(x_val, y_val, rbfi, lib=True):
    xnew = np.linspace(0.001, 2, 100)  # 100 values from 0.001 to 2
    if lib:
        fval = rbfi(xnew)
        plt.figure(3)
        plt.scatter(x_val, y_val, c='r', marker='o')
        plt.plot(xnew, fval, label="gaussian")
        plt.title('Rbf interpolation')
        plt.ylabel('Rbf')
        plt.legend()
        plt.show()
    else:
        fval = []
        for i in range(100):
            fval.append(rbf.s(xnew[i]))
        plt.figure(3)
        plt.scatter(x_val, y_val, c='r', marker='o')
        plt.plot(xnew, fval, label="gaussian")
        plt.title('Rbf Ciabo')
        plt.ylabel('Rbf')
        plt.legend()
        plt.show()


def evaluate(learning_rate, x_train, y_train, x_test, y_test):
    # load json and create model in this way we will use always the same model and weights
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    opt = optimizers.Adam(lr=learning_rate)  # default decay=0
    loaded_model.compile(optimizer=opt,
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
    # training the model and saving metrics in history
    epochs = 3
    history = loaded_model.fit(x_train, y_train, epochs=epochs, verbose=2, callbacks=[learningRateTracker()])

    results = loaded_model.evaluate(x_test, y_test)  # return loss and precision
    print("Loss, precision: ", results)

    return results[0]


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

# finds first three results
points = np.array([0.03078195,  0.1213984, 0.018049096])
values = np.array([])
for i in range(0, points.size):
     values = np.append(values, evaluate(points[i], x_train, y_train, x_test, y_test))

print("Learning rate: ", points)
print("Loss value: ", values)

for i in range(0, 2):
    rbf = r.RBF(points, values)
    rbf.interpolate()
    if i % 3 == 0:
        newx = np.power(10,-rbf.newxGivenf(-1000))
    else:
        newx = rbf.newxGivenf(0)
    points = np.append(points, newx)
    newf = evaluate(newx, x_train, y_train, x_test, y_test)
    values = np.append(values, newf)

minIndex = np.where(values == np.amin(values))
bestLearningRate = points[minIndex]
print("Best Learning rate found : ", bestLearningRate)
print("Loss value: ", values[minIndex])
print(points)
print(values)
plot_rbf(rbf.getX(), values, rbf, False)
'''
x = np.array([0.3, 1.4, 2])
f = np.array([2.3, 14.1, 13.3])
rbfi = interpolate.Rbf(x, f, function="gaussian")
rbf = r.RBF(x, f)
rbf.interpolate()
lambd = rbf.getMultipliers()
print(rbf.g(0))
print(rbf.g(1))
#plot_rbf(x, f, rbfi, True)
#plot_rbf(x, f, rbf, False)
v=rbf.newxGivenf(0)
print("aaaaaaaaaaaaaaa: ",v)
'''
