# ==================================================================================================
#                                       GLOBAL IMPORTS
# ==================================================================================================

import os
import sys
import time
import threading

# ==================================================================================================
#                                        LOCAL IMPORTS
# ==================================================================================================

current_dir = os.path.dirname(os.path.realpath(__file__))
os_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

utility_dir = os_dir + r'/utility'
controls_dir = current_dir + r'/controls'
sensors_dir = current_dir + r'/sensors'
camera_dir = sensors_dir + r'/camera'
drive_dir = current_dir + r'/drive'

sys.path.append(utility_dir)
sys.path.append(controls_dir)
sys.path.append(camera_dir)
sys.path.append(drive_dir)

from logger import *
