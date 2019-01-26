# --------------------------------------------------------------------------------------------------
# Global Imports 
# --------------------------------------------------------------------------------------------------

import os
import time
import serial
import sys

# --------------------------------------------------------------------------------------------------
# Local Imports 
# --------------------------------------------------------------------------------------------------

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

logger_dir = parent_dir + r'/logger'
sys.path.append(logger_dir) 
from logger import *

# --------------------------------------------------------------------------------------------------
# Main Class 
# --------------------------------------------------------------------------------------------------

class Serial:

    def __init__(self, port, baudrate=9600):

        self.port = serial.Serial(port=port, baudrate=9600, timeout=.1)

    def write(self, msg):

        logger.debug("Sent msg: {} to port: {}".format(msg,self.port))

        self.port.write(b'{}'.format(msg))


# --------------------------------------------------------------------------------------------------
# Sample Code
# --------------------------------------------------------------------------------------------------

if __name__ == '__main__':

   arduino_serial = Serial(port='COM13')

   while True:

   		arduino_serial.write('1234')

   		time.sleep(1)
