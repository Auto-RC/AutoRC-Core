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
#                                               I2C
# ==================================================================================================

class I2c:

    OFFSET = 0

    def __init__(self, address, bus):

        self.address = address
        self.bus = smbus.SMBus(bus)

    def read(self):

        raw_byte = self.bus.read_i2c_block_data(self.address,self.OFFSET,4)
        # byte = str(bin(raw_byte)[2:])

        print(raw_byte)

        # return byte

# ==================================================================================================
#                                            TEST CODE
# ==================================================================================================





