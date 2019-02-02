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

sys.path.append(utility_dir)
sys.path.append(controls_dir)
sys.path.append(camera_dir)

from logger import *
from controller import Controller
from memory import Memory
from iris import Iris

# ==================================================================================================
#                                           AutoRC
# ==================================================================================================

class AutoRC(threading.Thread):

    # ----------------------------------------------------------------------------------------------
    #                                           Initialize
    # ----------------------------------------------------------------------------------------------

    def __init__(self, controller_update_ms):

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
        self.controller.run()

        # Initializing flags
        # ------------------------------------------------------------------------------------------
        self.enable_vehicle = False
        self.enable_iris = False

    # ----------------------------------------------------------------------------------------------
    #                                        Core Functionality
    # ----------------------------------------------------------------------------------------------

    def toggle_vehicle(self):

        if self.enable_vehicle == False:
            self.enable_vehicle = True
            logger.debug("Vehicle enabled.")
        elif self.enable_vehicle == True:
            self.enable_vehicle = False
            logger.debug("Vehicle disabled.")

    def toggle_iris(self):

        if (self.enable_iris == False) and (not self.iris):

            self.iris = Iris(20, (128, 96), 'rgb')
            self.iris.run()

            self.enable_iris = True
            logger.debug("Iris enabled")

        elif (self.enable_iris == True) and (self.iris):

            self.iris.stop()
            del self.iris

            self.enable_iris = False
            logger.debug("Iris disabled")

    # ----------------------------------------------------------------------------------------------
    #                                               Run
    # ----------------------------------------------------------------------------------------------

    def run(self):

        while True:
            logger.debug("Button o: ", controller.ctrl_btn_val['O'])
            logger.debug("Button ^: ", controller.ctrl_btn_val['^'])
            if self.controller.ctrl_btn_val['O'] == True:
                self.toggle_vehicle()
            if self.controller.ctrl_btn_val['^'] == True:
                self.toggle_iris()


# ==================================================================================================
#                                            TEST CODE
# ==================================================================================================

if __name__ == '__main__':

    logger.setLevel(logging.DEBUG)

    instance = AutoRC(controller_update_ms=10)

    instance.run()