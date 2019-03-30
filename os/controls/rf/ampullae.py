# ==================================================================================================
#                                       GLOBAL IMPORTS
# ==================================================================================================

import smbus
import os
import sys
import time
from threading import Thread


# ==================================================================================================
#                                        LOCAL IMPORTS
# ==================================================================================================

current_dir = os.path.dirname(os.path.realpath(__file__))
sensors_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
parent_dir = os.path.abspath(os.path.join(sensors_dir, os.pardir))

utility_dir = parent_dir + r'/utility'
controls_dir = current_dir + r'/controls'

sys.path.append(utility_dir)
sys.path.append(controls_dir)

from logger import *
from i2c import I2c

# ==================================================================================================
#                                           Ampullae
# ==================================================================================================

class Ampullae(Thread):

    ADDRESS = 0x04
    BUS = 0

    def __init__(self, update_interval_ms):

        # Thread parameters
        # ------------------------------------------------------------------------------------------
        self.thread_name = "Ampullae"
        Thread.__init__(self, name=self.thread_name)

        # Main parameters
        # ------------------------------------------------------------------------------------------
        self.update_interval_ms = update_interval_ms

        self.i2c = I2c(self.ADDRESS,self.BUS)

        self.throttle = 0
        self.steering = 0
        self.mode = 0

        self.enable_i2c = True

    def run(self):

        logger.info("Controller thread started...")

        while self.enable_i2c == True:

            self.read()
            time.sleep(self.update_interval_ms / 1000)


    def read(self):

        raw_byte = self.i2c.read()
        byte = raw_byte

        for i in range(0,8-len(raw_byte)):
            byte = "0" + byte

        self.decode(byte)

    def decode(self, byte):

        logger.info("Byte: {}".format(byte))

        self.throttle = 0 # Throttle zero
        self.steering = 96 # Middle steering
        self.swb = 191 # Lower position
        self.swc = 255 # Lower position

        type = byte[0:2]
        value = int(byte[2:len(byte)])

        if type == "00":
            self.throttle = value
        elif type == "01":
            self.steering = value
        elif type == "10":
            self.swb = value
        elif type == "11":
            self.swc = value

        logger.info("Throttle: {}".format(self.throttle))

    def disable(self):

        self.enable_i2c = False

        logger.info("Ampullae thread ended")



# ==================================================================================================
#                                            TEST CODE
# ==================================================================================================

if __name__ == '__main__':

    ampullae = Ampullae(update_interval_ms=100)
    ampullae.run()


