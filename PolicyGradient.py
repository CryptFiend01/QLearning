from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
import keras.backend as K
from QEnv import *
import numpy as np
import random
import time
import os

BOARD_SIZE = 6

class PolicyNet:
    def __init__(self, actcnt, learning=0.001) -> None:
        self.actcnt = actcnt
        self.lr = learning
        self.wfname = 'maze_policy.h5'

    def createDNN(self):
        self.pmodel = Sequential([
            Flatten(input_shape=(BOARD_SIZE, BOARD_SIZE, 1)),
            Dense(256, activation='relu'),
            Dense(512, activation="relu"),
            Dense(self.actcnt, activation="softmax")
        ])

        self.pmodel.summary() 

        # def pg_loss(rets):
        #     def modified_crossentropy(action, action_probs):
        #         cost = K.categorical_crossentropy(action, action_probs, from_logits=False, axis=1 * rets)
        #         return K.mean(cost)
        #     return modified_crossentropy
        # self.pmodel.compile(optimizer=Adam(learning_rate=self.lr), loss=pg_loss)

        self.pmodel.compile(optimizer=Adam(learning_rate=self.lr), loss="categorical_crossentropy")

    def getAction(self, s):
        X = s / 3.0
        p = self.pmodel.predict(X, verbose=0)
        # return np.random.choice(np.argsort(p[0])[:3])
        # return np.argmax(p[0])
        # print(p.flatten())
        return np.random.choice(self.actcnt, 1, p=p.flatten())[0]

    def learning(self, s, a, r):
        X = np.reshape(s, (-1, BOARD_SIZE, BOARD_SIZE, 1))
        X = X / 3.0
        Y = np.eye(self.actcnt)[a] * np.reshape(r, (-1, 1))
        print(Y[:10])
        h = self.pmodel.fit(X, Y, batch_size=16, verbose=0)
        return h.history["loss"]

    def save(self):
        if os.path.exists(self.wfname):
            os.rename(self.wfname, self.wfname[:-3] + '_' + time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())) + ".h5")
        self.pmodel.save(self.wfname)

    def load(self):
        if os.path.exists(self.wfname):
            self.pmodel.load_weights(self.wfname)

class PolicyGradient:
    def __init__(self, env : QEnv) -> None:
        self.env = env
        self.net = PolicyNet(len(env.getAllActions()))
        self.net.createDNN()
        self.states = []
        self.actions = []
        self.rewards = []
        self.episode = 1000

    def getState(self):
        s = self.env.getStateNp()
        s = np.reshape(s, (-1, BOARD_SIZE, BOARD_SIZE, 1))
        return s

    def train(self):
        env = self.env
        
        for i in range(self.episode):
            env.reset()
            s = self.getState()
            rewards = 0
            while True:
                a = self.net.getAction(s)
                s1, r, done, _ = env.step(a)
                self.states.append(s)
                self.actions.append(a)
                self.rewards.append(r)
                rewards += r
                s = np.reshape(s1, (-1, BOARD_SIZE, BOARD_SIZE, 1))
                if done:
                    break

            for k in range(len(self.rewards) - 2, -1, -1):
                self.rewards[k] += 0.99 * self.rewards[k+1]
            mr = np.mean(self.rewards)
            stdr = np.std(self.rewards)
            sr = np.sum(self.rewards)
            print(f"reward sum: {sr} mean: {mr} std: {stdr}")
            for k in range(len(self.rewards)):
                self.rewards[k] = (self.rewards[k] - mr) / stdr
            loss = self.net.learning(self.states, self.actions, self.rewards)
            print(f"[{i}] loss: {loss} rewards: {rewards} step count: {len(self.actions)}")
            if i > 0 and i % 100 == 0:
                self.net.save()
            self.states, self.actions, self.rewards = [], [], []

    def play(self):
        env = self.env
        env.reset()

        self.net.load()

        rewards = 0
        while True:
            s = self.getState()
            a = self.net.getAction(s)
            _, r, done, _ = env.step(a)
            rewards += r
            if done:
                print(f"total reward: {rewards}")
                break

if __name__ == '__main__':
    pg = PolicyGradient(QEnv())
    try:
        pg.train()
    except KeyboardInterrupt:
        print("quit.")