import pygame
import os

class Controller:
    def __init__(self):
        self.axes = [3, 4, 0, 1, 2, 5]
        self.vals = []
        pygame.init()
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.display.init()
        screen = pygame.display.set_mode((1, 1))

    def return_vals(self):
        for event in pygame.event.get():
            pass

        joystick = pygame.joystick.Joystick(0)
        joystick.init()

        for i in range(len(self.axes)):
            self.vals[i] = joystick.get_axis(self.axes[i])

        return self.vals

