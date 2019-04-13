


#Test CODE FOR IMU



import board
import busio
import adafruit_lsm9ds1
import time

i2c = busio.I2C(board.SCL, board.SDA)
sensor = adafruit_lsm9ds1.LSM9DS1_I2C(i2c)
while True:
    print("a:", round(list(sensor.acceleration)[0],3))
    #print("g:", *sensor.gyro)
    time.sleep(0.001)

