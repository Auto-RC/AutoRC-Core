# ==================================================================================================
#                                       GLOBAL IMPORTS
# ==================================================================================================

import serial
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

class Ser:

    OFFSET = 0

    def __init__(self, baudrate, timeout):

        self.baudrate = baudrate
        self.timeout = timeout
        self.port = serial.Serial(port='/dev/ttyUSB0', baudrate=self.baudrate, timeout=self.timeout)

    def read(self):

        return self.port.read(size=8)

# ==================================================================================================
#                                            TEST CODE
# ==================================================================================================
