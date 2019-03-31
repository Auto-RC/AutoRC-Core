# ==================================================================================================
#                                           GLOBAL IMPORTS
# ==================================================================================================

import threading
import time
import os
import sys

# ==================================================================================================
#                                           LOCAL IMPORTS
# ==================================================================================================



# ==================================================================================================
#                                               DRIVE
# ==================================================================================================

class Drive(threading.Thread):

    THR_MAX = 64
    THR_MIN = 0

    STR_MAX = 64
    STR_MIN = 0

    def __init__(self, update_interval_ms, controller, pca9685):

        # Thread parameters
        # ------------------------------------------------------------------------------------------
        self.thread_name = "Drive"
        threading.Thread.__init__(self, name=self.thread_name)

        self.enable_drive = True
        self.update_interval_ms = update_interval_ms

        self.controller = controller
        self.pca9685 = pca9685

        self.steering = 0
        self.throttle = 0

        time.sleep(0.5)

    def run(self):

        print("Drive thread started")

        while self.enable_drive == True:

            # Getting values from controller
            # --------------------------------------------------------------------------------------
            self.compute_controls()

            self.pca9685.set_steering(self.steering)
            self.pca9685.set_throttle(self.throttle)

            time.sleep(self.update_interval_ms/1000)

    def disable(self):

        self.enable_drive = False

        self.controller.throttle = -1
        self.controller.brake = -1

        self.pca9685.set_steering(self.steering)
        self.pca9685.set_throttle(self.throttle)



    def compute_controls(self):

        if self.enable_drive == True:

            self.throttle = self.controller.thr/65
            self.steering = (self.controller.str-32)/65

# ==================================================================================================
#                                            TEST CODE
# ==================================================================================================
