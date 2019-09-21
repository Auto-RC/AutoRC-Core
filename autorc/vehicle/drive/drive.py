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

    def __init__(self, cerebellum, pca9685, update_interval_ms=10):

        # Thread parameters
        self.thread_name = "Drive"
        threading.Thread.__init__(self, name=self.thread_name)

        self.enabled = False
        self.update_interval_ms = update_interval_ms

        self.cerebellum = cerebellum
        self.pca9685 = pca9685

        self.steering = 0
        self.throttle = 0

        time.sleep(0.5)

    def get_frame(self):

        return [self.steering, self.throttle]

    def run(self):

        while True:

            if self.enabled == True:

                # Getting values from controller
                # --------------------------------------------------------------------------------------
                self.update_controls()

                self.pca9685.set_steering(self.steering)
                self.pca9685.set_throttle(self.throttle)

            time.sleep(self.update_interval_ms / 1000)

    def enable(self):

        self.enabled = True

    def disable(self):

        self.enabled = False

        self.throttle = -1
        self.steering = 0

        self.pca9685.set_steering(self.steering)
        self.pca9685.set_throttle(self.throttle)

    def update_controls(self):

        if self.enabled == True:

            self.throttle = self.cerebellum.thr
            self.steering = self.cerebellum.str
