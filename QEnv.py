from maze import *

class QEnv:
    def __init__(self):
        self.maze = MazeGame()
        self.actions = [LEFT, RIGHT, UP, DOWN]

    def reset(self):
        self.maze.reset()

    def action(self, a):
        res = self.maze.moveIt(a)
        done = False
        s = str(self.maze.pos)
        if self.maze.isWin():
            done = True
            s = 'done'
        
        return s, res, done

    def step(self, a):
        res = self.maze.moveIt(self.actions[a])
        done = self.maze.isWin()
        s = self.getStateNp()
        p = self.maze.pos
        return s, res, done, p

    def getState(self):
        return str(self.maze.pos)

    def getStateNp(self):
        d = self.maze.maze.copy()
        d[self.maze.pos] = 3
        return d

    def getAllActions(self):
        return self.actions

    def show(self):
        self.maze.show()

    def close(self):
        pass