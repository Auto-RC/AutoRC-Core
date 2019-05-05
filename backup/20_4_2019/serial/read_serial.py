import serial
import sys

if (len(sys.argv) == 3):
    path = "/dev/" + str(sys.argv[1])
    b_rate = sys.argv[2]
else:
    path = "/dev/ttyUSB0"
    b_rate = 9600
print("path: ", path, "  b_rate: ", b_rate)

ser = serial.Serial(path, b_rate, timeout=1)
ser.baudrate = b_rate

while 1:
    m = ser.readline()
    if ser.readline() != b'':
        print(ser.readline())