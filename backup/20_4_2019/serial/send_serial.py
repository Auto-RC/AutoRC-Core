import serial
import sys
from controller import Controller
import time

if (len(sys.argv) == 4):
    path = "/dev/" + str(sys.argv[1])
    b_rate = sys.argv[2]
else:
    path = "/dev/ttyUSB0"
    b_rate = 250000
print("path: ", path, "  b_rate: ", b_rate)

ser = serial.Serial(path, b_rate, timeout=1)
ser.baudrate = b_rate

controller = Controller()


def send(forward, side):
    forward = int(((forward + 1) * 100) + 900)
    side = int(side * 200) + 900
    print(forward, side)
    for i in range(4 - len(str(side))):
        ser.write(b'0')
    ser.write(str(side).encode())
    for i in range(4 - len(str(forward))):
        ser.write(b'0')
    ser.write(str(forward).encode())
    time.sleep(0.05)


while(1):
    controller.update()
    send(controller.throttle, controller.steering)