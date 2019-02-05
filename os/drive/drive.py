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

    def __init__(self, update_interval_ms, controller, pca9685):

        # Thread parameters
        # ------------------------------------------------------------------------------------------
        self.thread_name = "Drive"
        threading.Thread.__init__(self, name=self.thread_name)

        self.enable_drive = True
        self.update_interval_ms = update_interval_ms

        self.controller = controller
        self.pca9685 = pca9685

        self.safety_enable = True
        time.sleep(0.5)

    def run(self):

        print("Drive thread started")

        while self.enable_drive == True:

            # Getting values from controller
            # --------------------------------------------------------------------------------------
            self.controller.throttle = self.controller.ctrl_axis_val['r_t']
            self.controller.brake = self.controller.ctrl_axis_val['l_t']
            self.controller.steering = self.controller.ctrl_axis_val['r_j_x']

            if (self.safety_enable == True) and \
               (self.controller.throttle != 0) and \
               (self.controller.brake != 0):
                self.safety_enable = False

            if self.safety_enable == True:
                self.controller.throttle = -1
                self.controller.brake = -1

            self.pca9685.set_steering(self.controller.steering)
            self.pca9685.set_throttle(self.compute_throttle(self.controller.throttle, self.controller.brake))

            time.sleep(self.update_interval_ms/1000)

    def disable(self):

        self.enable_drive = False

    def compute_throttle(self,
                         throttle,  # Input Range [-1,1]
                         brake):  # Input Range [-1,1]

        if brake > -0.8:
            return -(brake + 1) / 2  # Output range [0,-1]
        else:
            return (throttle + 1) / 2  # Output range [0,1]


# ==================================================================================================
#                                            TEST CODE
# ==================================================================================================
