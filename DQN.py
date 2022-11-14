from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import plot_model

class DeepQLearning:
    def __init__(self, env) -> None:
        self.env = env
        self.model = Sequential()

    def createDNN(self):
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))

    def train(self):
        pass

    def play(self):
        pass

def train_for_pic(x_train, y_train, x_test, y_test):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))

    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr = 0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd)

    model.fit(x_train, y_train, batch_size=32, epochs=10)
    model.evaluate(x_test, y_test, batch_size=32)
    plot_model(model, show_shapes=True)
    return model


#mod = train_for_pic()
