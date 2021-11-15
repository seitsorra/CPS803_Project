import util
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import metrics

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.layers import Dropout
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler
from keras.callbacks import History
from keras.optimizers import Adam


filename = "../data/train_cleaned.csv"

data = util.load(filename)

train_x, train_y = data[0]
valid_x, valid_y = data[1]
test_x, test_y = data[2]

print(train_x.shape)
print(valid_x.shape)
print(test_x.shape)

def train_random_forest_classifier():
    clf = RandomForestClassifier(max_depth=25, min_samples_split=2)
    clf.fit(train_x, train_y)

    print(clf.score(train_x, train_y))
    print(clf.score(test_x, test_y))

def train_keras():
    model = Sequential()
    model.add(Dense(24, input_dim=train_x.shape[1], activation='relu'))
    model.add(Dense(4, activation='relu',kernel_regularizer=l2(0.2), bias_regularizer=l2(0.01)))
    model.add(Dense(4, activation='relu',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.001)))
    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    opt = Adam(learning_rate=0.005)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    # fit the keras model on the dataset
    lr_model_history = model.fit(train_x, train_y, epochs=10000, batch_size=100, validation_data=(test_x, test_y))
    # evaluate the keras model
    _, accuracy = model.evaluate(test_x, test_y)
    print('Accuracy: %.2f' % (accuracy*100))

    # Plot the loss function
    fig, ax = plt.subplots(1, 1, figsize=(6,4))
    ax.plot(np.sqrt(lr_model_history.history['loss']), 'r', label='train')
    ax.plot(np.sqrt(lr_model_history.history['val_loss']), 'b' ,label='val')
    ax.set_xlabel(r'Epoch', fontsize=20)
    ax.set_ylabel(r'Loss', fontsize=20)
    ax.legend()
    ax.tick_params(labelsize=20)

    # Plot the accuracy
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(np.sqrt(lr_model_history.history['accuracy']), 'r', label='train')
    ax.plot(np.sqrt(lr_model_history.history['val_accuracy']), 'b' ,label='val')
    ax.set_xlabel(r'Epoch', fontsize=20)
    ax.set_ylabel(r'Accuracy', fontsize=20)
    ax.legend()
    ax.tick_params(labelsize=20)
    plt.show()


def logistic_regression_model():
    solvers = ['saga']
    for s in solvers:
        clf = LogisticRegression(max_iter=100000,solver=s, multi_class='multinomial', penalty='elasticnet', l1_ratio=0.7)
        clf.fit(np.concatenate((train_x, valid_x), axis=0), np.concatenate((train_y, valid_y), axis=0))
        #clf.predict(test_x)
        # Perform 6-fold cross validation
        scores = cross_val_score(clf, train_x, train_y, cv=10)
        print ("Cross-validated scores:", scores)

        predictions = cross_val_predict(clf, test_x, test_y, cv=2)
        accuracy = metrics.r2_score(test_y, predictions)
        print("Cross-Predicted Accuracy:", accuracy)

        print(f"{s} accuracy on training: {clf.score(train_x,train_y)} valid: {clf.score(valid_x, valid_y)}")
        # print(clf.score(train_x,train_y))
        print(clf.score(test_x, test_y))
    
def sgd_model():
    epochs=1000
    learning_rate = 0.05
    decay_rate = learning_rate / epochs
    momentum = 0.8

    num_classes = 1
    sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

    # build the model
    input_dim = train_x.shape[1]

    lr_model = Sequential()
    lr_model.add(Dense(64, activation='relu', kernel_initializer='uniform', input_dim = input_dim)) 
    lr_model.add(Dropout(0.1))
    lr_model.add(Dense(64, kernel_initializer='uniform', activation='relu'))
    lr_model.add(Dense(num_classes, kernel_initializer='uniform', activation='softmax'))

    # compile the model
    lr_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['acc'])

    # Fit the model
    batch_size = int(input_dim/100)

    x = np.concatenate((train_x, valid_x), axis=0)
    y = np.concatenate((train_y, valid_y), axis=0)

    lr_model_history = lr_model.fit(x, y,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(test_x, test_y))
    
    _, accuracy = lr_model.evaluate(test_x, test_y)
    print('Accuracy: %.2f' % (accuracy*100))
    
    # # Plot the loss function
    # fig, ax = plt.subplots(1, 1, figsize=(10,6))
    # ax.plot(np.sqrt(lr_model_history.history['loss']), 'r', label='train')
    # ax.plot(np.sqrt(lr_model_history.history['val_loss']), 'b' ,label='val')
    # ax.set_xlabel(r'Epoch', fontsize=20)
    # ax.set_ylabel(r'Loss', fontsize=20)
    # ax.legend()
    # ax.tick_params(labelsize=20)

    # # Plot the accuracy
    # fig, ax = plt.subplots(1, 1, figsize=(10,6))
    # ax.plot(np.sqrt(lr_model_history.history['acc']), 'r', label='train')
    # ax.plot(np.sqrt(lr_model_history.history['val_acc']), 'b' ,label='val')
    # ax.set_xlabel(r'Epoch', fontsize=20)
    # ax.set_ylabel(r'Accuracy', fontsize=20)
    # ax.legend()
    # ax.tick_params(labelsize=20)
    # plt.show()
    
def sgd_model_1():
    # solution
    epochs = 1000
    learning_rate = 0.1 # initial learning rate
    decay_rate = 0.1
    momentum = 0.8

    # define the optimizer function
    sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

    input_dim = train_x.shape[1]
    num_classes = 1
    batch_size = 10

    # build the model
    exponential_decay_model = Sequential()
    exponential_decay_model.add(Dense(4, activation='relu', kernel_initializer='uniform', input_dim = input_dim)) 
    #exponential_decay_model.add(Dropout(0.1))
    exponential_decay_model.add(Dense(4, kernel_initializer='uniform', activation='relu'))
    exponential_decay_model.add(Dense(num_classes, kernel_initializer='uniform', activation='sigmoid'))

    # compile the model
    exponential_decay_model.compile(loss='binary_crossentropy', 
                                    optimizer=sgd, 
                                    metrics=['acc'])
                                    
    # define the learning rate change 
    def exp_decay(epoch):
        lrate = learning_rate * np.exp(-decay_rate*epoch)
        return lrate
        
    # learning schedule callback
    loss_history = History()
    lr_rate = LearningRateScheduler(exp_decay)
    callbacks_list = [loss_history, lr_rate]

    # you invoke the LearningRateScheduler during the .fit() phase
    x = np.concatenate((train_x, valid_x), axis=0)
    y = np.concatenate((train_y, valid_y), axis=0)

    exponential_decay_model_history = exponential_decay_model.fit(x, y,
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        callbacks=callbacks_list,
                                        verbose=1,
                                        validation_data=(test_x, test_y))
    _, accuracy = exponential_decay_model.evaluate(test_x, test_y)
    print('Accuracy: %.2f' % (accuracy*100))


if __name__ == '__main__':
    train_keras()