import pygame


axes = [3, 4, 0, 1, 2, 5]
vals = []



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



    print(vals)