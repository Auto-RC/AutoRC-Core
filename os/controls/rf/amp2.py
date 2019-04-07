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
from ser import Ser

# ==================================================================================================
#                                           Ampullae
# ==================================================================================================

class Ampullae(Thread):

    port = '/dev/ttyUSB0'

    def __init__(self, baudrate, timeout, update_interval_ms):

        # Thread parameters
        # ------------------------------------------------------------------------------------------
        self.thread_name = "Ampullae"
        Thread.__init__(self, name=self.thread_name)

        # Main parameters
        # ------------------------------------------------------------------------------------------
        self.baudrate = baudrate
        self.timeout = timeout
        self.update_interval_ms = update_interval_ms

        self.ser = Ser(self.baudrate, self.timeout)

        self.enable_ser = True


    def run(self):

        logger.info("Controller thread started...")

        while self.enable_ser == True:

            self.read()
            time.sleep(self.update_interval_ms / 1000)


    def read(self):

        raw = str(self.ser.read())
        raw = raw.replace('b','')
        raw = raw.replace("'",'')
        print(len(raw))
        print(raw[0:1])
        self.thr = int(raw[0:1])
        self.str = int(raw[2:3])
        self.swb = int(raw[4:5])
        self.swc = int(raw[6:7])

        logger.info("THR {} STR {} SWB: {} SWC: {} ".format(self.thr, self.str, self.swb, self.swc))

    def disable(self):

        self.enable_ser = False

        logger.info("Ampullae thread ended")



# ==================================================================================================
#                                            TEST CODE
# ==================================================================================================

if __name__ == '__main__':

    ampullae = Ampullae(9600, 0.01, update_interval_ms=50)
    ampullae.run()


