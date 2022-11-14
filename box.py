import sys

RED = 1
BLUE = 2

CAMP_NAME = ['', 'RED', 'BLUE']

class DotBox:
    def __init__(self):
        self.board = [
            -1, 0, -1, 0, -1, 0, -1,
            0, 0, 0, 0, 0, 0, 0,
            -1, 0, -1, 0, -1, 0, -1,
            0, 0, 0, 0, 0, 0, 0,
            -1, 0, -1, 0, -1, 0, -1,
            0, 0, 0, 0, 0, 0, 0,
            -1, 0, -1, 0, -1, 0, -1]
        self.grids = [8, 10, 12, 22, 24, 26, 36, 38, 40]
        self.width = 7
        self.height = 7
        self.colored_num = 0
        self.camp_num = [0, 0]
        self.next_color = RED

    def getWidth(self):
        return self.width

    def getHeight(self):
        return self.height

    def getNextColor(self):
        return self.next_color

    def getXY(self, i):
        return int(i % self.width), int(i / self.width)

    def getLeft(self, i):
        x, y = self.getXY(i)
        if x == 0:
            return -1
        return i - 1

    def getLeftColor(self, i):
        k = self.getLeft(i)
        if k >= 0:
            return self.board[k]
        else:
            return -1

    def getRight(self, i):
        x, y = self.getXY(i)
        if x >= self.width - 1:
            return -1
        return i + 1

    def getRightColor(self, i):
        k = self.getRight(i)
        if k >= 0:
            return self.board[k]
        else:
            return -1

    def getUp(self, i):
        x, y = self.getXY(i)
        if y == 0:
            return -1
        return i - self.width

    def getUpColor(self, i):
        k = self.getUp(i)
        if k >= 0:
            return self.board[k]
        else:
            return -1

    def getDown(self, i):
        x, y = self.getXY(i)
        if y >= self.height - 1:
            return -1
        return i + self.width

    def getDownColor(self, i):
        k = self.getDown(i)
        if k >= 0:
            return self.board[k]
        else:
            return -1

    def isEdgeVertical(self, i):
        x, y = self.getXY(i)
        return y % 2 == 1

    def fillGrid(self, i, color):
        if i == -1 or i not in self.grids:
            return
        if self.board[i] != 0:
            return
        if self.getLeftColor(i) > 0 and \
            self.getRightColor(i) > 0 and \
            self.getUpColor(i) > 0 and \
            self.getDownColor(i) > 0:
            self.board[i] = color
            self.colored_num += 1
            self.camp_num[color - 1] += 1

    def setEdge(self, i):
        color = self.next_color
        if i < 0 or i >= len(self.board):
            return -1
        if i in self.grids:
            return -1
        if self.board[i] != 0:
            return -1

        self.board[i] = color
        
        colored_num = self.colored_num
        if self.isEdgeVertical(i):
            self.fillGrid(self.getLeft(i), color)
            self.fillGrid(self.getRight(i), color)
        else:
            self.fillGrid(self.getUp(i), color)
            self.fillGrid(self.getDown(i), color)
        if colored_num == self.colored_num:
            if self.next_color == RED:
                self.next_color = BLUE
            else:
                self.next_color = RED
        return 0

    def isFinsih(self):
        return self.colored_num >= len(self.grids)

    def getWinner(self):
        if not self.isFinsih():
            return 0
        if self.camp_num[RED - 1] > self.camp_num[BLUE - 1]:
            return RED
        else:
            return BLUE

    def getBoard(self):
        return self.board

    def showBoard(self):
        sg = [''] * 3
        def setcol(s, c):
            for i in range(len(s)):
                s[i] += c

        def setgrid(sg, c):
            for i in range(len(sg)):
                if i == int(len(sg) / 2):
                    sg[i] += '  ' + c + '  '
                else:
                    sg[i] += '     '

        s = ''
        for i in range(len(self.board)):
            x, y = self.getXY(i)
            if y % 2 == 0:
                if self.board[i] == -1:
                    s += '+'
                elif self.board[i] == 0:
                    s += '-----'
                elif self.board[i] == RED:
                    s += 'rrrrr'
                elif self.board[i] == BLUE:
                    s += 'bbbbb'
                else:
                    print('error board[%d]: %d' % (i, self.board[i]))
                if x == self.width - 1:
                    print(s)
                    s = ''
            else:
                if x % 2 == 0:
                    if self.board[i] == 0:
                        setcol(sg, '|')
                    elif self.board[i] == RED:
                        setcol(sg, 'r')
                    elif self.board[i] == BLUE:
                        setcol(sg, 'b')
                    else:
                        print('error board[%d]: %d' % (i, self.board[i]))
                else:
                    if self.board[i] == 0:
                        setgrid(sg, ' ')
                    elif self.board[i] == RED:
                        setgrid(sg, 'R')
                    elif self.board[i] == BLUE:
                        setgrid(sg, 'B')
                    else:
                        print('error board[%d]: %d' % (i, self.board[i]))
                if x == self.width - 1:
                    for line in sg:
                        print(line)
                    sg = [''] * 3
        if self.isFinsih():
            print(CAMP_NAME[self.getWinner()] + ' is win!!!')

def my_input(promt):
    if sys.version_info.major == 2:
        return raw_input(promt)
    else:
        return input(promt)

if __name__ == '__main__':
    db = DotBox()
    db.showBoard()
    while True:
        color = db.getNextColor()
        s = my_input('Please enter pos(like |,1,1 or -,1,1) [%s]:' % (CAMP_NAME[color]))
        p = s.split(',')
        if len(p) < 3:
            continue
        x, y = 0, 0
        if p[0] == '|':
            y = (int(p[1]) - 1) * 2 + 1
            x = (int(p[2]) - 1) * 2
        elif p[0] == '-':
            y = (int(p[1]) - 1) * 2
            x = (int(p[2]) - 1) * 2 + 1
        else:
            continue
        d = x + y * db.getWidth()
        if db.setEdge(d) != 0:
            print('Input error, ignore it!')
            continue
        db.showBoard()
        if db.isFinsih():
            break
            
