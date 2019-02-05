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
utility_dir = current_dir + r'/utility'
controls_dir = current_dir + r'/controls'
sensors_dir = current_dir + r'/sensors'
camera_dir = sensors_dir + r'/camera'
drive_dir = current_dir + r'/drive'

sys.path.append(utility_dir)
sys.path.append(controls_dir)
sys.path.append(camera_dir)
sys.path.append(drive_dir)

from logger import *
from controller import Controller
from memory import Memory
from iris import Iris
from pca_9685 import PCA9685
from drive import Drive

# ==================================================================================================
#                                           AutoRC
# ==================================================================================================

class AutoRC(threading.Thread):

    # ----------------------------------------------------------------------------------------------
    #                                           Initialize
    # ----------------------------------------------------------------------------------------------

    def __init__(self, controller_update_ms):

        logger.debug("Initial")

        # Thread parameters
        # ------------------------------------------------------------------------------------------
        self.thread_name = "AutoRC"
        threading.Thread.__init__(self, name=self.thread_name)

        # Initializing parameters
        # ------------------------------------------------------------------------------------------
        self.controller_update_ms = controller_update_ms

        # Initializing controller
        # ------------------------------------------------------------------------------------------
        self.controller = Controller(wait_interval_ms = self.controller_update_ms)
        self.controller.start()

        # Initializing PCA9685 driver
        # ------------------------------------------------------------------------------------------
        self.pca9685 = PCA9685()

        # Initializing flags
        # ------------------------------------------------------------------------------------------
        self.enable_vehicle = False
        self.enable_iris = False

    # ----------------------------------------------------------------------------------------------
    #                                        Core Functionality
    # ----------------------------------------------------------------------------------------------

    def toggle_vehicle(self):

        if self.enable_vehicle == False:

            self.drive = Drive(update_interval_ms = 10,
                               controller = self.controller,
                               pca9685=self.pca9685)

            self.drive.start()

            self.enable_vehicle = True
            logger.debug("Vehicle enabled.")

        elif self.enable_vehicle == True:

            self.drive.disable()
            del self.drive

            self.enable_vehicle = False
            logger.debug("Vehicle disabled.")

    def toggle_iris(self):

        if (self.enable_iris == False): # and (not self.iris):

            self.iris = Iris(20, (128, 96), 'rgb')
            self.iris.run()

            self.enable_iris = True
            logger.debug("Iris enabled")

        elif (self.enable_iris == True): # and (self.iris):

            self.iris.stop()
            del self.iris

            self.enable_iris = False
            logger.debug("Iris disabled")

    # ----------------------------------------------------------------------------------------------
    #                                               Run
    # ----------------------------------------------------------------------------------------------

    def run(self):

        logger.debug("AutoRC started")

        while True:
            logger.debug("VEH: {} IRIS: {}".format(self.enable_vehicle,self.enable_iris))

            if self.controller.ctrl_btn_val['O'] == True:
                self.toggle_vehicle()
            if self.controller.ctrl_btn_val['^'] == True:
                self.toggle_iris()

            time.sleep(100/1000)

        # logger.debug("AutoRC exited")


# ==================================================================================================
#                                            TEST CODE
# ==================================================================================================

if __name__ == '__main__':

    logger.setLevel(logging.DEBUG)

    instance = AutoRC(controller_update_ms=100)

    instance.run()