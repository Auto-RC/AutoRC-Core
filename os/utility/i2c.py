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


# ==================================================================================================
#                                           I2C
# ==================================================================================================

class I2c:

    def __init__(self, address):

        # Main parameters
        # ------------------------------------------------------------------------------------------
        self.address = address

        self.bus = smbus.SMBus(0)


    def read(self):
        n = self.bus.read_byte(self.address)
        n_binary = bin(n)[2:]

        return n_binary



# ==================================================================================================
#                                            TEST CODE
# ==================================================================================================





