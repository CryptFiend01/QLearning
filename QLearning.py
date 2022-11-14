import numpy as np
import pandas as pd
from QEnv import *
from QCarEnv import *
import os

class QTable:
    def __init__(self, actions, learning_rate=0.7, reward_decay=0.1, greedy=0.9):
        self.qtable = pd.DataFrame(columns=actions, dtype=np.float64)
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        self.greedy = greedy
        self.actions = actions

    def chooseAction(self, s):
        self.checkState(s)

        if np.random.uniform() < self.greedy:
            action_rewards = self.qtable.loc[s, :]
            a = np.random.choice(action_rewards[action_rewards == np.max(action_rewards)].index)
        else:
            a = np.random.choice(self.actions)
        q = self.qtable.loc[s, a]
        return a, q

    def chooseActionMax(self, s):
        self.checkState(s)
        action_rewards = self.qtable.loc[s, :]
        a = np.random.choice(action_rewards[action_rewards == np.max(action_rewards)].index)
        q = self.qtable.loc[s, a]
        return a, q

    def learn(self, s, a, r, s1):
        self.checkState(s1)

        if r < 0:
            self.qtable.loc[s, a] = r
            return

        predict = self.qtable.loc[s, a]
        if s1 != 'done':
            target = r + self.reward_decay * self.qtable.loc[s1, :].max()
        else:
            target = r
        self.qtable.loc[s, a] += self.learning_rate * (target - predict)

    def checkState(self, s):
        if s not in self.qtable.index:
            self.qtable = self.qtable.append(
                pd.Series(
                    [0] * len(self.actions), 
                    index=self.qtable.columns, 
                    name=s))

    def show(self):
        print(self.qtable)

class QLearning:
    def __init__(self, env):
        self.qtable = QTable(env.getAllActions())
        self.env = env

    def train(self):
        env = self.env
        for i in range(1000):
            env.reset()
            s = env.getState()

            while True:
                a, q = self.qtable.chooseAction(s)
                #print("state %s choose action %s qvalue %f" % (s, a, q))
                s1, r, done = env.action(a)
                self.qtable.learn(s, a, r, s1)
                s = s1
                if done:
                    break
            if (i + 1) % 10 == 0:
                print("learning processing %d%%" % ((i + 1) / 10))

    def play(self):
        env = self.env
        env.reset()
        s = env.getState()

        is_win = False
        while True:
            env.show()
            a, q = self.qtable.chooseActionMax(s)
            print("Choose action " + str(a) + " Q(s,a)=" + str(q))
            s1, r, done = env.action(a)
            s = s1
            if done:
                is_win = r > 0
                break

        def Result(is_win):
            if is_win:
                return 'win!!!'
            else:
                return 'lose /_\\'
        env.show()
        print('Result is :', Result(is_win))
        self.qtable.show()
        env.close()

if __name__ == '__main__':
    env = QEnv()
    q = QLearning(env)
    q.train()
    print(" ")
    q.play()
    os.system("pause")

    