import pygame
import math
import sys

class Car:
    def __init__(self):
        self.img = None
        self.rect = None
        self.angle = 270
        self.realImg = None
        self.speed = 5
        self.pos = [0, 0]
        self.realRect = None

    def Init(self):
        self.img = pygame.image.load("car1.png")
        self.Reset()

    def GetRect(self):
        return self.realRect

    def RotateLeft(self):
        self.angle += 5
        if self.angle > 360:
            self.angle -= 360
        self.realImg = pygame.transform.rotate(self.img, self.angle)

    def RotateRight(self):
        self.angle -= 5
        if self.angle < 0:
            self.angle += 360
        self.realImg = pygame.transform.rotate(self.img, self.angle)

    def MoveForward(self):
        self.pos[0] += -self.speed * math.sin(self.angle * math.pi / 180)
        self.pos[1] += -self.speed * math.cos(self.angle * math.pi / 180)
        self.rect.x = self.pos[0]
        self.rect.y = self.pos[1]

    def draw(self, screen):
        screen.blit(self.realImg, self.rect)
        pygame.draw.rect(screen, (0,255,0), self.rect, 1)
        rect = self.realImg.get_rect()
        rect.x = self.rect.x
        rect.y = self.rect.y
        self.realRect = rect
        pygame.draw.rect(screen, (255,0,0), rect, 1)
        pygame.draw.circle(screen, (255, 255, 0), [rect.centerx, rect.centery], rect.width / 2, 1)

    def Reset(self):
        self.angle = 270
        self.rect = self.img.get_rect()
        self.realImg = pygame.transform.rotate(self.img, self.angle)
        self.pos[0] = self.rect.x
        self.pos[1] = self.rect.y + 25
        self.rect.y = self.pos[1]
        self.realRect = self.realImg.get_rect()
        self.realRect.x = self.rect.x
        self.realRect.y = self.rect.y

    def GetState(self):
        return str(self.rect.x) + '_' + str(self.rect.y) + '_' + str(self.angle)

class RaceMap:
    def __init__(self):
        self.blocks = []
        self.win_rect = [700, 420, 800, 520]

    def Init(self):
        self.blocks.append([0, 0, 640, 20])
        self.blocks.append([0, 100, 480, 20])
        self.blocks.append([640, 0, 20, 400])
        self.blocks.append([460, 120, 20, 420])
        self.blocks.append([640, 400, 200, 20])
        self.blocks.append([460, 540, 340, 20])

    def draw(self, screen):
        for b in self.blocks:
            pygame.draw.rect(screen, (100, 200, 100), b, 0)

    def IsInserect(self, rect):
        for b in self.blocks:
            if rect.left > b[0] + b[2] or rect.top > b[1] + b[3] or \
                rect.right < b[0] or rect.bottom < b[1]:
                continue
            else:
                return True
        return False

    def IsWin(self, rect):
        x, y = rect.centerx, rect.centery
        if x >= self.win_rect[0] and x <= self.win_rect[2] and y >= self.win_rect[1] and y <= self.win_rect[3]:
            return True
        return False

class RaceGame:
    def __init__(self):
        self.width = 800
        self.height = 600
        self.screen = None
        self.car = None
        self.map = None
        self.status = "playing"
        self.reward = 0.0

    def Init(self):
        pygame.init()
        sz = self.width, self.height
        self.screen = pygame.display.set_mode(sz)
        self.car = Car()
        self.car.Init()
        self.map = RaceMap()
        self.map.Init()
        self.clock = pygame.time.Clock()
        self.reward = 1.0

        self.win_txt = self.createText("WIN  \(^o^)/~", 30, (120, 220, 120))
        self.lose_txt = self.createText("LOSE  o(T_T)o", 30, (220, 120, 120))

    def createText(self, txt, font_size, color):
        font = pygame.font.SysFont("Monaco", font_size)
        return font.render(txt, True, color)

    def GetState(self):
        return self.car.GetState()

    def Reset(self):
        self.car.Reset()
        self.status = "playing"

    def draw(self):
        self.screen.fill((10, 10, 10))
        if self.status == "playing":
            self.map.draw(self.screen)
            self.car.draw(self.screen)
        elif self.status == "lose":
            self.screen.blit(self.lose_txt, ((self.width - self.lose_txt.get_width()) // 2, (self.height - self.lose_txt.get_height()) // 2))
        elif self.status == "win":
            self.screen.blit(self.win_txt, ((self.width - self.win_txt.get_width()) // 2, (self.height - self.win_txt.get_height()) // 2))
        pygame.display.flip()

    def Update(self):
        running = True
        while running:
            self.clock.tick(10)
            for evt in pygame.event.get():
                if evt.type == pygame.QUIT:
                    running = False
                    break
                elif evt.type == pygame.KEYDOWN and self.status != "playing":
                    if evt.key == pygame.K_SPACE:
                        self.Reset()

            if self.status == "playing":
                pressed = pygame.key.get_pressed()
                if pressed[pygame.K_LEFT]:
                    self.car.RotateLeft()
                elif pressed[pygame.K_RIGHT]:
                    self.car.RotateRight()
                self.car.MoveForward()
                if self.map.IsInserect(self.car.GetRect()):
                    self.status = "lose"
                elif self.map.IsWin(self.car.GetRect()):
                    self.status = "win"
            self.draw()
        pygame.quit()

    def ComputeRun(self, action):
        #self.clock.tick(10)
        if self.reward < 60:
            self.reward += 0.01
        reward = self.reward
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                pygame.quit()
                return
        if self.status == "playing":
            if action == "LEFT":
                self.car.RotateLeft()
            elif action == "RIGHT":
                self.car.RotateRight()
            self.car.MoveForward()
            if self.map.IsInserect(self.car.GetRect()):
                self.status = "lose"
                reward = -1000
            elif self.map.IsWin(self.car.GetRect()):
                self.status = "win"
                reward = 100
            self.draw()
        return reward

    def WaitClose(self):
        running = True
        while running:
            for evt in pygame.event.get():
                if evt.type == pygame.QUIT:
                    running = False
                    break
        pygame.quit()

if __name__ == '__main__':
    race = RaceGame()
    race.Init()
    race.Update()