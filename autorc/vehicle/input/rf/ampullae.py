# ==================================================================================================
#                                       GLOBAL IMPORTS
# ==================================================================================================

import smbus
import os
import sys
import time
from threading import Thread
import logging

# ==================================================================================================
#                                            LOGGER SETUP
# ==================================================================================================

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(module)s %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger.setLevel(logging.INFO)

# ==================================================================================================
#                                        LOCAL IMPORTS
# ==================================================================================================

current_dir = os.path.dirname(os.path.realpath(__file__))
sensors_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
parent_dir = os.path.abspath(os.path.join(sensors_dir, os.pardir))

utility_dir = parent_dir + r'/utility'
controls_dir = current_dir + r'/input'

sys.path.append(utility_dir)
sys.path.append(controls_dir)
logger.info(utility_dir)

from srl import Srl

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

        self.srl = Srl(self.baudrate, self.timeout)

        self.enable_srl = True
        self.thr = 10
        self.str = 55
        self.swc = 99
        self.swb = 99


    def run(self):

        logger.info("Controller thread started...")

        while self.enable_srl == True:

            self.read()
            self.compute_drive()
            time.sleep(self.update_interval_ms / 1000)

    def read(self):

        raw = str(self.srl.read())
        raw = raw.replace('b','')
        raw = raw.replace("'",'')
        raw = raw.replace("/", '')
        raw = raw.replace(r"\\", '')
        raw = raw.replace("x", '')

        if len(raw) == 8:
            self.thr_raw = int(raw[0:2])
            self.str_raw = int(raw[2:4])
            self.swc = int(raw[4:6])
            self.swb = int(raw[6:8])

            logger.debug("THR {} STR {} SWB: {} SWC: {} ".format(self.thr, self.str, self.swb, self.swc))

    def compute_drive(self):

        self.thr = (self.thr_raw - 10) / 90
        self.str = (self.str_raw - 55) / 45

    def disable(self):

        self.enable_srl = False
        logger.info("Ampullae thread ended")



# ==================================================================================================
#                                            TEST CODE
# ==================================================================================================

if __name__ == '__main__':

    ampullae = Ampullae(9600, 0.01, update_interval_ms=20)
    ampullae.run()


