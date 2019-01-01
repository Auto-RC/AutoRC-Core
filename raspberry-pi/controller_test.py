import pygame
import os
import time

axes = [3, 4, 2]
vals = [0] * len(axes)



pygame.init()
os.environ["SDL_VIDEODRIVER"] = "dummy"
pygame.display.init()
screen = pygame.display.set_mode((1,1))
while(1):
    for event in pygame.event.get():
        pass

    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    for i in range(len(axes)):
        vals[i] = joystick.get_axis(axes[i])

    print("throttle: ", vals[1], "   brake: ", vals[0], "   steering: ", vals[2])
    time.sleep(1)


