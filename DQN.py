from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, Input
from keras.optimizers import SGD
from keras.utils import plot_model
from QEnv import *
from collections import deque
import numpy as np
import random

class DeepQNet:
    def __init__(self, actions, learning_rate=0.01, decay=1e-6) -> None:
        self.actions = actions
        self.model = None
        self.learning_rate = learning_rate
        self.decay = decay
        self.isLearned = False

    def createDNN(self):
        actions = self.actions

        x = network = Input(shape=(6, 6, 1))
        network = Conv2D(16, (2, 2), activation='relu', padding='same')(network)
        network = Conv2D(32, (2, 2), activation='relu', padding='same')(network)
        polnet = Flatten()(network)
        polnet = Dense(512, activation='relu')(polnet)
        polnet = Dense(len(actions), activation='softmax')(polnet)

        valnet = Flatten()(network)
        valnet = Dense(512, activation='relu')(valnet)
        valnet = Dense(1, activation='tanh')(valnet)
        self.model = Model(inputs=x, outputs=[polnet, valnet])

        sgd = SGD(lr = self.learning_rate, decay=self.decay, momentum=0.9, nesterov=True)
        self.model.compile(loss="categorical_crossentropy", optimizer=sgd)

    def getAction(self, s):
        # print(s.shape)
        if self.isLearned:
            p = self.model.predict(s, verbose=0)
            i = np.argmax(p[0][0])
            a = self.actions[i]
            q = p[0][0]
            return a, q
        else:
            i = random.randint(0, len(self.actions) - 1)
            a = self.actions[i]
            q = 1.0 / len(self.actions)
            return a, np.array([q] * len(self.actions))

    def learning(self, s, a, r):
        states = np.array(s)
        states = states.reshape(-1, 6, 6, 1)
        actprobs = np.array(a)
        actprobs = actprobs.reshape(-1, len(self.actions), 1)
        rewards = np.array(r)
        rewards = rewards.reshape(-1, 1)
        print(f'training states: {states.shape} actprobs: {actprobs.shape} rewards: {rewards.shape}')
        # loss = self.model.evaluate(states, [actprobs, rewards], batch_size=len(states))
        self.model.fit(states, [actprobs, rewards], batch_size=len(states), epochs=10)
        self.isLearned = True

class DQLearning:
    def __init__(self, env : QEnv) -> None:
        self.env = env
        self.net = DeepQNet(env.getAllActions())
        self.net.createDNN()
        self.epoch = 1000
        self.buffer = deque(maxlen=1000)

    def playBatch(self):
        n = 0
        env = self.env
        env.reset()
        states = []
        actprobs = []
        rewards = []
        while True:
            s = env.getStateNp()
            s = np.array(s)
            s = s.reshape(-1, 6, 6, 1)
            a, q = self.net.getAction(s)
            _, r, done = self.env.action(a)
            states.append(s)
            actprobs.append(q)
            rewards.append(r)
            if done:
                break
            n += 1
        return zip(states, actprobs, rewards)

    def learn(self):
        batch = random.sample(self.buffer, 500)
        s = [d[0] for d in batch]
        a = [d[1] for d in batch]
        r = [d[2] for d in batch]
        self.net.learning(s, a, r)

    def train(self):
        for i in range(10000):
            results = self.playBatch()
            self.buffer.extend(results)
            if i > 0 and i % 500 == 0:
                self.learn()

    def play(self):
        env = self.env
        env.reset()

        while True:
            s = env.getStateNp()
            s = np.array(s)
            s = s.reshape(-1, 6, 6, 1)
            a, _ = self.net.getAction(s)
            _, _, done = env.action(a)
            if done:
                break

if __name__ == '__main__':
    env = QEnv()
    dqn = DQLearning(env)
    dqn.train()
    dqn.play()
