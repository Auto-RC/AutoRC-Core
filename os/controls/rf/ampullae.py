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

        self.enable_i2c = True

        self.thr = 0  # Throttle zero
        self.str = 96  # Middle steering
        self.swb = 191  # Lower position
        self.swc = 255  # Lower position

    def run(self):

        logger.info("Controller thread started...")

        while self.enable_i2c == True:

            self.read()
            time.sleep(self.update_interval_ms / 1000)


    def read(self):

        raw_bytes = self.i2c.read()

        if raw_bytes[0]-192 > 0:
            self.thr = raw_bytes[0]-192
        if raw_bytes[1]-192 > 0:
            self.str = raw_bytes[1]-192

        self.swb = raw_bytes[2]
        self.swc = raw_bytes[3]

        logger.info("THR {} STR {} SWB: {} SWC: {} ".format(self.thr, self.str, self.swb, self.swc))

    def disable(self):

        self.enable_i2c = False

        logger.info("Ampullae thread ended")



# ==================================================================================================
#                                            TEST CODE
# ==================================================================================================

if __name__ == '__main__':

    ampullae = Ampullae(update_interval_ms=50)
    ampullae.run()


