import serial
import sys
import pygame

axes = [3, 4, 0, 1, 2, 5]
vals = []

if (len(sys.argv) == 4):
    path = "/dev/" + str(sys.argv[1])
    b_rate = sys.argv[2]
else:
    path = "/dev/ttyUSB0"
    b_rate = 250000
print("path: ", path, "  b_rate: ", b_rate)

ser = serial.Serial(path, b_rate, timeout=1)
ser.baudrate = b_rate

pygame.init()

while(1):
    for event in pygame.event.get():
        pass

    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    for i in range(len(axes)):
        vals[i] = joystick.get_axis(axes[i])

    print(vals)