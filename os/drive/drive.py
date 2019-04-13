# ==================================================================================================
#                                           GLOBAL IMPORTS
# ==================================================================================================

import threading
import time
import os
import sys
import logging

# ==================================================================================================
#                                            LOGGER SETUP
# ==================================================================================================

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(module)s %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger.setLevel(logging.INFO)

# ==================================================================================================
#                                               DRIVE
# ==================================================================================================

class Drive(threading.Thread):

    THR_MAX = 64
    THR_MIN = 0

    STR_MAX = 64
    STR_MIN = 0

    def __init__(self, controller, pca9685, update_interval_ms=10):

        # Thread parameters
        self.thread_name = "Drive"
        threading.Thread.__init__(self, name=self.thread_name)
        self._stop_event = threading.Event()

        self.enabled = False
        self.update_interval_ms = update_interval_ms

        self.controller = controller
        self.pca9685 = pca9685

        self.steering = 0
        self.throttle = 0

        time.sleep(0.5)

    def run(self):

        logger.info("Drive thread started")

        self.enabled = True
        while self.enabled == True:

            # Getting values from controller
            # --------------------------------------------------------------------------------------
            self.compute_controls()

            self.pca9685.set_steering(self.steering)
            self.pca9685.set_throttle(self.throttle)

            time.sleep(self.update_interval_ms/1000)

        logger.info("Drive thread stopped")

    def disable(self):

        self.enabled = False

        self.controller.throttle = -1
        self.controller.brake = -1

        self.pca9685.set_steering(self.steering)
        self.pca9685.set_throttle(self.throttle)

        self._stop_event.set()

    def compute_controls(self):

        if self.enabled == True:

            self.throttle = (self.controller.thr - 10) / 90
            self.steering = (self.controller.str - 55) / 45

    def stopped(self):

        return self._stop_event.is_set()

# ==================================================================================================
#                                            UNIT TEST
# ==================================================================================================
