
def createMaze():
    maze = [0,0,0,0,0,0,
            0,0,1,1,1,0,
            0,0,1,2,0,0,
            0,0,1,0,0,0,
            0,0,0,0,0,0]
    return maze

LEFT = 1
RIGHT = 2
UP = 3
DOWN = 4

DIR_NAME = ['', 'LEFT', 'RIGHT', 'UP', 'DOWN']

class MazeGame:
    def __init__(self):
        self.maze = createMaze()
        self.pos = 0
        self.width = 6
        self.height = len(self.maze) / 6

    def reset(self):
        self.pos = 0

    def moveIt(self, d):
        #global LEFT, RIGHT, UP, DOWN, DIR_NAME
        if self.isWin():
            return -1
        x = self.pos % self.width
        y = self.pos / self.width
        #print("From(%d,%d) move %s" % (x, y, DIR_NAME[d]))
        pos = self.pos
        if d == LEFT:
            if x > 0:
                pos -= 1
            else:
                return -1
        elif d == RIGHT:
            if x < self.width - 1:
                pos += 1
            else:
                return -1
        elif d == UP:
            if y > 0:
                pos -= self.width
            else:
                return -1
        elif d == DOWN:
            if y < self.height - 1:
                pos += self.width
            else:
                return -1

        if self.maze[pos] != 1:
            self.pos = pos
        else:
            return -1

        #print("New pos: " + str(self.pos))
            
        if self.maze[self.pos] == 2:
            return 1
        else:
            return 0

    def isWin(self):
        return self.maze[self.pos] == 2

    def show(self):
        if self.isWin():
            print('==========')
            print('|  WIN!  |')
            print('==========')
            return
        print('----------\n')
        s = ''
        for i in range(len(self.maze)):
            if self.pos == i:
                s += '*'
            else:
                s += str(self.maze[i])
            s += ' '
            if (i + 1) % self.width == 0:
                s += '\n'
        print(s)
        print('----------')

if __name__ == '__main__':
    mz = MazeGame()
    mz.show()
    while True:
        try:
            d = int(input('move dir(L(1), R(2), U(3), D(4)):'))
        except:
            continue
        mz.moveIt(d)
        mz.show()
        if mz.isWin():
            break
