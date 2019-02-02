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

sys.path.append(utility_dir)
sys.path.append(controls_dir)

from logger import *
from controller import Controller

# ==================================================================================================
#                                           AutoRC
# ==================================================================================================

class AutoRC(threading.Thread):

    def __init__(self, controller_update_ms):

        # Thread parameters
        # ------------------------------------------------------------------------------------------
        self.thread_name = "AutoRC"
        threading.Thread.__init__(self, name=self.thread_name)

        # Initializing parameters
        # ------------------------------------------------------------------------------------------
        self.controller_update_ms = controller_update_ms

        # Initializing objects
        # ------------------------------------------------------------------------------------------
        self.controller = Controller(wait_interval_ms = self.controller_update_ms)


    def run_manual(self):

        self.controller.run()

    def camera_test(self):
        with picamera.PiCamera() as camera:
            camera.resolution = (128, 96)
            camera.start_preview()
            # Camera warm-up time
            time.sleep(2)
            for i in range(200):
                camera.capture('track1/pic{}.jpg'.format(i))
