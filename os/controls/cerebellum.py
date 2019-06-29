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


# ==================================================================================================
#                                             CORTEX
# ==================================================================================================

class Cerebellum(threading.Thread):

    def __init__(self, controller, cortex, corti, update_interval_ms):

        self.controller = controller
        self.cortex = cortex
        self.corti = corti
        self.update_interval_ms = update_interval_ms

        # Thread parameters
        self.thread_name = "Cerebellum"
        threading.Thread.__init__(self, name=self.thread_name)

        self.auto = False

        self.thr = 10
        self.str = 55

        self.state = dict()
        self.state['angles']    = None
        self.state['midpoints'] = None
        self.state['x_accel']   = None
        self.state['y_accel']   = None
        self.state['z_accel']   = None

    def update_state(self):

        self.state['angles']    = self.cortex.angles
        self.state['midpoints'] = self.cortex.midpoints
        # self.state['x_accel']   = state['x_accel']
        # self.state['y_accel']   = state['y_accel']
        # self.state['z_accel']   = state['z_accel']

    def compute_controls(self):

        none = [False, False, False]
        not_none = 0

        avg_angle = 0

        for i in range(3):
            if self.state['angles'][i] is None:
                none[i] = True
            else:
                avg_angle += self.state['angles'][i]
                not_none += 1

        avg_angle /= not_none

        # print(avg_angle)
        scaled_angle = (avg_angle/90) * 45 + 55

        self.thr = self.controller.thr
        self.str = scaled_angle

    def run(self):

        while True:

            if self.auto == False:
                self.thr = self.controller.thr
                self.str = self.controller.str
            elif self.auto == True:
                self.update_state()
                self.compute_controls()

            time.sleep(self.update_interval_ms / 1000)

# ------------------------------------------------------------------------------
#                                 SAMPLE CODE
# ------------------------------------------------------------------------------

if __name__ == '__main__':

    retina = Retina()

    retina.load_npy(file_name='/Users/arnavgupta/car_data/raw_npy/oculus-2019-06-16 20;49;28.264824.npy')
    retina.test_line_detection()





