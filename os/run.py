# ==================================================================================================
#                                       GLOBAL IMPORTS
# ==================================================================================================

import os
import sys
import time
import threading
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
utility_dir = current_dir + r'/utility'
controls_dir = current_dir + r'/input/rf'
sensors_dir = current_dir + r'/sensors'
camera_dir = sensors_dir + r'/camera'
imu_dir = sensors_dir + r'/imu'
drive_dir = current_dir + r'/drive'
cortex_dir = current_dir + r'/cortex'

sys.path.append(utility_dir)
sys.path.append(controls_dir)
sys.path.append(camera_dir)
sys.path.append(drive_dir)
sys.path.append(imu_dir)
sys.path.append(cortex_dir)

from memory import Memory
from oculus import Oculus
from pca_9685 import PCA9685
from drive import Drive
from memory import Memory
from ampullae import Ampullae
from corti import Corti
from cortex import Cortex

# ==================================================================================================
#                                           AutoRC
# ==================================================================================================

class AutoRC(threading.Thread):

    # ----------------------------------------------------------------------------------------------
    #                                           Initialize
    # ----------------------------------------------------------------------------------------------

    def __init__(self):

        # Thread parameters
        # ------------------------------------------------------------------------------------------
        self.thread_name = "AutoRC"
        threading.Thread.__init__(self, name=self.thread_name)

        # Initializing controller
        # ------------------------------------------------------------------------------------------
        self.controller = Ampullae(baudrate = 9600, timeout = 0.01, update_interval_ms = 10)
        self.controller.start()

        # Initializing PCA9685 driver
        # ------------------------------------------------------------------------------------------
        self.pca9685 = PCA9685()

        # Initializing array of running modules
        # ------------------------------------------------------------------------------------------
        self.modules = []

        # Initializing flags
        # ------------------------------------------------------------------------------------------
        self.enable_vehicle = False
        self.enable_oculus = True
        self.enable_memory = False
        self.enable_corti = False
        self.enable_cortex = False

        # Initializing modules
        # ------------------------------------------------------------------------------------------
        self.drive = Drive(controller=self.controller, pca9685=self.pca9685,update_interval_ms=10)
        self.drive.start()

        self.corti = Corti(update_interval_ms=50)
        self.corti.start()

        self.oculus = Oculus(20, (128, 96), 'rgb')
        self.oculus.run()
        self.modules.append('oculus')

        self.cortex = Cortex(update_interval_ms=50,oculus=self.oculus)

    # ----------------------------------------------------------------------------------------------
    #                                        Core Functionality
    # ----------------------------------------------------------------------------------------------

    def toggle_vehicle(self):

        if self.enable_vehicle == False:

            self.drive.enable()

            self.enable_vehicle = True
            logger.debug("Vehicle enabled.")

            self.modules.append('drive')

        elif self.enable_vehicle == True:

            self.drive.disable()

            self.enable_vehicle = False
            logger.debug("Vehicle disabled.")

            self.modules.remove('drive')

    def toggle_oculus(self):

        if (self.enable_oculus == False): # and (not self.oculus):

            self.oculus.run()

            self.enable_oculus = True
            logger.debug("Oculus enabled")

            # self.modules.append('oculus')

        elif (self.enable_oculus == True): # and (self.oculus):

            self.oculus.disable()

            self.enable_oculus = False
            logger.debug("Oculus disabled")

            # self.modules.remove('oculus')

    def toggle_memory(self):

        if (self.enable_memory == False):

            self.enable_memory = True
            self.memory = Memory(self.modules)

            logger.debug("Started capturing data")
            logger.debug("Storing memory from {}".format(self.modules))

        elif (self.enable_memory == True):

            self.memory.save()
            del self.memory

            self.enable_memory = False
            logger.debug("Stopped capturing data")

    def toggle_corti(self):

        if (self.enable_corti == False):

            self.corti.enable()

            self.enable_corti = True
            logger.debug("Started Corti...")

            self.modules.append('corti')

        elif (self.enable_corti == True):

            self.corti.disable()

            self.enable_corti = False
            logger.debug("Stopped Corti.")

            self.modules.remove('corti')

    def toggle_cortex(self):

        if (self.enable_cortex == False):

            self.cortex.enable()

            self.enable_cortex = True
            logger.debug("Started Cortex...")

        elif (self.enable_cortex == True):

            self.cortex.disable()

            self.enable_cortex = False
            logger.debug("Stopped Cortex")

    def add_data_packet(self):

        data_packet = dict()

        if 'oculus' in self.modules:

            picture = self.oculus.get_frame()
            data_packet['oculus'] = picture

        if 'drive' in self.modules:

            steering = self.drive.steering
            throttle = self.drive.throttle
            data_packet['drive'] = [steering, throttle]

        if 'corti' in self.modules:

            acceleration = self.corti.acceleration
            data_packet['corti'] = [acceleration]

        self.memory.add(data_packet)

    # ----------------------------------------------------------------------------------------------
    #                                               Run
    # ----------------------------------------------------------------------------------------------

    def run(self):

        logger.debug("AutoRC live")

        while True:


            logger.info("VEH: {} CORTI: {} OCULUS: {} MEM: {} CORTEX: {} THR: {} STR: {} SWB: {} SWC: {}".format(self.enable_vehicle,self.enable_corti, self.enable_oculus,self.enable_memory, self.enable_cortex, self.controller.thr, self.controller.str, self.controller.swb, self.controller.swc))

            if self.enable_memory:
                self.add_data_packet()

            if (self.controller.swb > 50) and (self.enable_vehicle == False):
                self.toggle_vehicle()
                self.toggle_corti()
            elif(self.controller.swb < 50) and (self.enable_vehicle == True):
                self.toggle_vehicle()
                self.toggle_corti()

            if (self.controller.swc > 20) and (self.enable_cortex == True):
                self.toggle_cortex()
            elif (self.controller.swc < 20) and (self.enable_cortex == False):
                self.toggle_cortex()

            if (self.controller.swc < 70) and (self.controller.swc > 20) and (self.enable_memory == False):
                self.toggle_memory()
            elif ( (self.controller.swc < 70) or (self.controller.swc > 20) ) and (self.enable_memory == True):
                self.toggle_memory()

            time.sleep(100/1000)


# ==================================================================================================
#                                            TEST CODE
# ==================================================================================================

if __name__ == '__main__':

    logger.setLevel(logging.DEBUG)

    instance = AutoRC()

    instance.run()