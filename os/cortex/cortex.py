# ==================================================================================================
#                                          GLOBAL IMPORTS
# ==================================================================================================

import os
import sys
import time
import logging
import threading

# ==================================================================================================
#                                             LOGGING
# ==================================================================================================

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(module)s %(levelname)s: %(message)s',
                            datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger.setLevel(logging.INFO)

# ==================================================================================================
#                                          LOCAL IMPORTS
# ==================================================================================================

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
vision_dir = parent_dir + r'/vision'

sys.path.append(vision_dir)

from retina_backup import Retina

# ==================================================================================================
#                                             CORTEX
# ==================================================================================================

class Cortex(threading.Thread):

    def __init__(self, update_interval_ms, oculus):

        self.oculus = oculus
        self.retina = Retina()

        # Thread parameters
        self.thread_name = "Cortex"
        threading.Thread.__init__(self, name=self.thread_name)

        self.enabled = False
        self.update_interval_ms = update_interval_ms

    def process_vision(self):

        # Setting the current frame
        frame = self.oculus.get_frame()
        self.retina.set_frame(frame)

        # Filtering color
        self.retina.filter_colors(lower_rgb_range=[0,45,60],upper_rgb_range=[100,250,250])

        # Detecting lines
        self.lanes = self.retina.detect_lines()

        logger.info("Lane -> Angle: {} Midpoint: {}".format(self.lanes['angles'],self.lanes['midpoints']))

    def enable(self):

        self.enabled = True

    def disable(self):

        self.enabled = False

    def run(self):

        while True:

            if self.enabled == True:

                self.process_vision()

            time.sleep(self.update_interval_ms / 1000)

# ------------------------------------------------------------------------------
#                                 SAMPLE CODE
# ------------------------------------------------------------------------------

if __name__ == '__main__':

    retina = Retina()

    retina.load_npy(file_name='oculus-2019-04-19 18;16;06.500887.npy')
    retina.test_line_detection()





