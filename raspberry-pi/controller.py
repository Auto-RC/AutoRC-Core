import pygame
import os

class Controller:
    def __init__(self):
        self.axes = [3, 4, 2]
        self.throttle = -1
        self.brake = -1
        self.steering = 0
        pygame.init()
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.display.init()
        screen = pygame.display.set_mode((1, 1))

    def update(self):
        for event in pygame.event.get():
            pass

        joystick = pygame.joystick.Joystick(0)
        joystick.init()

        vals = [0] * len(self.axes)

        for i in range(len(self.axes)):
            vals[i] = joystick.get_axis(self.axes[i])

        self.throttle = vals[1]
        self.brake = vals[0]
        self.steering = vals[2]

