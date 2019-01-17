import pygame
import os
import time

class Controller:
    def __init__(self):
        self.axes = [3, 4, 2]
        self.buttons = [1, 12]
        self.throttle = -1
        self.brake = -1
        self.steering = 0
        self.on = True
        self.capturing = False

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

        vals = [0] * len(self.buttons)

        for i in range(len(self.buttons)):
            vals[i] = joystick.get_button(self.buttons[i])

        if vals[0]:
            print("x pressed")
            self.capturing = not self.capturing
            if self.capturing:
                print("capturing")
            else:
                print("saving")
            time.sleep(0.5)

        if vals[1]:
            self.on = not self.on
