from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import util
from keras.models import Sequential
from keras.layers import Dense

filename = "../data/train.csv"

data = util.load_data(filename)

train_x, train_y = data[0]
test_x, test_y = data[2]

def train_random_forest_classifier():
    clf = RandomForestClassifier(max_depth=25, min_samples_split=2)
    clf.fit(train_x, train_y)

    print(clf.score(train_x, train_y))
    print(clf.score(test_x, test_y))

def train_keras():
    model = Sequential()
    model.add(Dense(24, input_dim=41, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit the keras model on the dataset
    model.fit(train_x, train_y, epochs=10000, batch_size=10)
    # evaluate the keras model
    _, accuracy = model.evaluate(test_x, test_y)
    print('Accuracy: %.2f' % (accuracy*100))

if __name__ == '__main__':
    train_random_forest_classifier()