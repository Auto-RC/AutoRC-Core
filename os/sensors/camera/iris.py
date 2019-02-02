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
sensors_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
parent_dir = os.path.abspath(os.path.join(sensors_dir, os.pardir))

utility_dir = parent_dir + r'/utility'
controls_dir = current_dir + r'/controls'

sys.path.append(utility_dir)
sys.path.append(controls_dir)

from logger import *
from camera import PiCamera

# ==================================================================================================
#                                           IRIS
# ==================================================================================================

class Iris(threading.Thread):

    def __init__(self,
<<<<<<< HEAD
                 frame_rate=20,
                 resolution=(128, 96),
                 format='rgb'
                 ):
=======
                 frame_rate=20):


>>>>>>> 4308febb311cc156680a5a9a0d7d02f77db9f622

        logger.setLevel(logging.DEBUG)

        # Main parameters
        # ------------------------------------------------------------------------------------------
        self.frame_rate = frame_rate
        self.resolution = resolution
        self.format = format

        self.cam = PiCamera(self.resolution, self.frame_rate, self.format)

    def run(self):
        logger.info("Initializing controller thread...")

        t = Thread(target=self.cam.update, args=())
        t.daemon = True
        t.start()

        logger.info("Camera thread running")

    def get_current_picture(self):
        return self.cam.run_threaded()

    def stop(self):
        self.cam.shutdown()

        logger.info("Camera thread ended")


# ==================================================================================================
#                                            TEST CODE
# ==================================================================================================

if __name__ == '__main__':

    c = Iris(20, (128, 96), 'rgb')

    c.run()

    for i in range(10):
        print(c.get_current_picture())
        time.sleep(.5)

    c.stop()




