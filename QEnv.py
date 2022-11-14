from maze import *

class QEnv:
    def __init__(self):
        self.maze = MazeGame()

    def reset(self):
        self.maze.reset()

    def action(self, a):
        res = self.maze.moveIt(a)
        done = False
        s = str(self.maze.pos)
        if res != 0:
            done = True
            s = 'done'
        
        return s, res, done

    def getState(self):
        return str(self.maze.pos)

    def getAllActions(self):
        return [LEFT, RIGHT, UP, DOWN]

    def show(self):
        self.maze.show()

    def close(self):
        pass