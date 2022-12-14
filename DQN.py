from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, Input
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
from QEnv import *
from collections import deque
import numpy as np
import random
import os
import sys
from memory_profiler import profile

class DeepQNet:
    def __init__(self, actcnt, learning_rate=0.01, decay=0.9) -> None:
        self.actcnt = actcnt
        self.tmodel = None
        self.qmodel = None
        self.learning_rate = learning_rate
        self.decay = decay
        self.greedy = 1
        self.isLearned = False
        self.greedy_decay = 0.995
        self.greedy_min = 0.01
        self.learn_times = 0
        self.fix_times = 30

    def createDNN(self):
        # 决策网络，根据决策网络选择下一步行动，网络得出每种行动对应的奖励值
        self.qmodel = Sequential([
            Flatten(input_shape=(6, 6, 1)),
            Dense(36, activation='relu'),
            Dense(72, activation='relu'),
            Dense(self.actcnt, activation='linear')
        ])
        # self.qmodel.trainable = True
        self.qmodel.summary()

        # 目标网络，用于固定奖励值，慢于决策网络的更新，防止target一直在变导致学习不稳定
        self.tmodel = Sequential([
            Flatten(input_shape=(6, 6, 1)),
            Dense(36, activation='relu'),
            Dense(72, activation='relu'),
            Dense(self.actcnt, activation='linear')
        ])


        # sgd = SGD(learning_rate = self.learning_rate, decay=self.decay, momentum=0.9, nesterov=True)
        # sgd = Adam(learning_rate=self.learning_rate)
        # self.vmodel.compile(loss="categorical_crossentropy", optimizer=sgd)
        self.qmodel.compile(optimizer=RMSprop(learning_rate=self.learning_rate), loss="mse")

    def getAction(self, s):
        if self.isLearned and np.random.uniform() >= self.greedy:
            # print(s.shape)
            p = self.qmodel.predict(s, verbose=0)
            # print(p)
            a = np.argmax(p[0])
            return a
        else:
            a = random.randint(0, self.actcnt - 1)
            return a

    def learning(self, s, a, r, s1, d, n):
        losses = []
        for i in range(n):
            state = s[i].reshape(-1, 6, 6, 1)
            state1 = s1[i].reshape(-1, 6, 6, 1)
            y = self.qmodel.predict(state, verbose=0)
            q = self.tmodel.predict(state1, verbose=0)

            reward = r[i]
            action = a[i]
            done = d[i]
            target = reward
            if not done:
                target += self.decay * np.amax(q[0])
            y[0][action] = target
            h = self.qmodel.fit(state, y, verbose=0)
            loss = h.history['loss'][0]
            losses.append(loss)
        print(f"---------------> finish batch {n}")
        self.tmodel.set_weights(self.qmodel.get_weights())
        return losses

    def learningBatch(self, s, a, r, s1, d, n):
        states = s.reshape(-1, 6, 6, 1)
        states1 = s1.reshape(-1, 6, 6, 1)

        actions = a
        rewards = r
        #print(rewards[:10])
        #print(d[:10])
        # print(f'training states: {states.shape} actions: {actions.shape} rewards: {rewards.shape} state1: {states1.shape}')

        y = self.qmodel.predict(states, batch_size=16, verbose=0)
        q = self.tmodel.predict(states1, batch_size=16, verbose=0)
        # print(y.shape)
        # print(q.shape)
        # y = q * decay + r
        # print(f"states: {np.reshape(states[0], (1, 36))} actions: {actions[0]} y: {y[0]} q:{q[0]}")
        for i in range(n):
            # if rewards[i] > 100:
            #     print(f"train finish game reward is {rewards[i]}")
            target = rewards[i]
            if not d[i]:
                target += self.decay * np.amax(q[i])
            # print(f"i={i} action[i]={actions[i]} target={target}")
            y[i][actions[i]] = target
        # print(f"learn y: {y[0]}")
        h = self.qmodel.fit(states, y, batch_size=16, epochs=3, verbose=0)
        if self.greedy >= self.greedy_min:
            self.greedy *= self.greedy_decay
            # print(f"greedy-----> {self.greedy}")
        self.learn_times += 1
        if self.learn_times >= self.fix_times:
            self.learn_times = 0
            self.tmodel.set_weights(self.qmodel.get_weights())
        self.isLearned = True
        # print(h.history["loss"])
        return h.history["loss"][0]

    def save(self, fname):
        self.qmodel.save(fname)

    def load(self, fname):
        if os.path.exists(fname):
            print("load weight.")
            self.qmodel.load_weights(fname)
            self.isLearned = True
            self.greedy = self.greedy_min

        
class DQLearning:
    def __init__(self, env : QEnv) -> None:
        self.env = env
        self.net = DeepQNet(len(env.getAllActions()))
        self.net.createDNN()
        self.epoch = 1000
        self.batch = 64
        # self.bmaxlen = 128
        # self.bpos = 0
        self.buffer = deque(maxlen=128)
        
    def getState(self):
        s = np.array(env.getStateNp())
        s = s.reshape(-1, 6, 6, 1)
        return s

    def learn(self):
        if len(self.buffer) < self.batch:
            return
        batch = random.sample(self.buffer, self.batch)
        s = np.array([d[0] for d in batch])
        a = np.array([d[1] for d in batch])
        r = np.array([d[2] for d in batch], dtype=np.float64)
        s1 = np.array([d[3] for d in batch])
        d = np.array([d[4] for d in batch])
        # s, a, r, s1 = random.choice(self.buffer)
        return self.net.learningBatch(s, a, r, s1, d, self.batch)

    @profile(stream=open("mem.log", "w+"))
    def train(self):
        self.net.load("maze.h5")
        env = self.env
        
        try:
            for i in range(100):
                poscount = {}
                total_reward = 0
                steps = set()
                stepcount = 0
                # loss = -1
                env.reset()
                s = self.getState()
                buf = []
                while True:
                    a = self.net.getAction(s)
                    s1, r, done, p = self.env.step(a)
                    # print(f"a: {a} pos: ({int(p%6)}, {int(p/6)})")
                    steps.add(p)
                    stepcount += 1
                    s1 = np.array(s1)

                    if poscount.get(p):
                        poscount[p] += 1
                    else:
                        poscount[p] = 1
                    
                    self.buffer.append([s.reshape(6, 6, 1), a, r, s1.reshape(6, 6, 1), done])
                    loss = self.learn()
                    total_reward += r
                    s = s1.reshape(-1, 6, 6, 1)
                    if done:
                        print(f"total reward {total_reward} step count {stepcount} no repeat steps {len(steps)} buffer len {len(self.buffer)}")
                        # print(f"memory buffer size: {sys.getsizeof(self.buffer)} net size: {sys.getsizeof(self.net)} game size: {sys.getsizeof(self.env)}")
                        print(f"pos count: {poscount}")
                        if stepcount <= 10:
                            self.net.save("maze.h5")
                        # final_reward = 2500 - total_reward
                        # r = int(final_reward / stepcount)
                        # for b in buf:
                        #     b[2] += r
                        # self.buffer.extend(buf)
                        break
                
                if i % 10 == 0:
                    if isinstance(loss, list):
                        lossmax = np.amax(loss)
                        lossmin = np.amin(loss)
                        lossavg = np.average(loss)
                        print(f"process {i} reward {total_reward} lossmax {lossmax} lossmin {lossmin} lossavg {lossavg} buffer len {len(self.buffer)}")
                    else:
                        print(f"process {i} loss {loss} reward {total_reward} buffer len {len(self.buffer)}")

            # self.net.save("maze.h5")
        except KeyboardInterrupt:
            return
        

    def play(self):
        env = self.env
        env.reset()
        self.net.load("maze.h5")

        # print(env.getStateNp())
        steps = 0
        s = self.getState()

        try:
            while True:
                a = self.net.getAction(s)
                s1, _, done, _ = self.env.step(a)
                s = np.reshape(s1, (-1, 6, 6, 1))
                env.show()
                steps += 1
                if done:
                    break
        except KeyboardInterrupt:
            print("abort")
        print(f"use steps {steps}")

def testModel():
    model = Sequential([
            Flatten(input_shape=(6, 6, 1)),
            Dense(32, activation='relu'),
            Dense(64, activation='relu'),
            Dense(4, activation='linear')
        ])
    model.summary()
    model.compile(optimizer=RMSprop(learning_rate=0.001), loss="mse")

    env = QEnv()
    env.reset()
    s = env.getStateNp()
    s1, r, done = env.step(RIGHT-1)
    s = np.array(s)
    s = s.reshape(-1, 6, 6, 1)
    s1 = np.array(s1)
    s1 = s1.reshape(-1, 6, 6, 1)
    y = model.predict(s)
    q = model.predict(s1)
    print(f"s: {s.reshape(-1, 36)} s1: {s1.reshape(-1, 36)}")
    print(f"predict y: {y} q: {q}")
    y[0][1] = r + 0.5 * q[0][1]
    print(f"learn y: {y} reward: {r}")

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    env = QEnv()
    dqn = DQLearning(env)
    # dqn.train()
    dqn.play()
    # testModel()
    
