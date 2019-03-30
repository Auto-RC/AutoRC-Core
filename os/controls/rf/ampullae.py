# ==================================================================================================
#                                       GLOBAL IMPORTS
# ==================================================================================================

import smbus
import os
import sys
import time
from threading import Thread
from i2c import I2c

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

# ==================================================================================================
#                                           Ampullae
# ==================================================================================================

class Ampullae(Thread):

    def __init__(self, address, update_interval_ms):

        # Thread parameters
        # ------------------------------------------------------------------------------------------
        self.thread_name = "Ampullae"
        Thread.__init__(self, name=self.thread_name)

        # Main parameters
        # ------------------------------------------------------------------------------------------
        self.address = address
        self.update_interval_ms = update_interval_ms

        self.i2c = I2c(address)

        self.throttle = 0
        self.steering = 0
        self.mode = 0

        self.enable_i2c = True


    def run(self):
        logger.info("Drive thread started")

        while self.enable_i2c == True:

            self.read()

            time.sleep(self.update_interval_ms / 1000)


    def read(self):

        n_binary = self.i2c.read()

        logger.info("Arduino Input: {}".format(n_binary))

        if n_binary[0] == 0 and n_binary[1] == 1:
            self.throttle = int(n_binary[2:], 2)

        if n_binary[0] == 1 and n_binary[1] == 0:
            self.steering = int(n_binary[2:], 2)

        if n_binary[0] == 1 and n_binary[1] == 1:
            self.mode = int(n_binary[2:], 2)

        logger.info("Steering: {}".format(self.throttle))
        logger.info("Throttle: {}".format(self.steering))
        logger.info("Mode: {}".format(self.mode))

    def disable(self):

        self.enable_i2c = False

        logger.info("Ampullae thread ended")



# ==================================================================================================
#                                            TEST CODE
# ==================================================================================================

if __name__ == '__main__':

    ampullae = Ampullae(address=0x1e,update_interval_ms=100)
    ampullae.run()


