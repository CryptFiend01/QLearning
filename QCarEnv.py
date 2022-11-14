from race import *

class QCarEnv:
    def __init__(self):
        self.race = RaceGame()
        self.race.Init()

    def reset(self):
        self.race.Reset()

    def action(self, a):
        r = self.race.ComputeRun(a)
        s = self.race.GetState()
        done = (r < 0 or r >= 100)
        return s, r, done

    def getState(self):
        return self.race.GetState()

    def getAllActions(self):
        return ["LEFT", "RIGHT", "PASS"]

    def show(self):
        pass

    def close(self):
        self.race.WaitClose()