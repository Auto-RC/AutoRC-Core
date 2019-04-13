# ==================================================================================================
#                                       GLOBAL IMPORTS
# ==================================================================================================

import serial
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
logger.setLevel(logging.DEBUG)

# ==================================================================================================
#                                               I2C
# ==================================================================================================

class Srl:

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
