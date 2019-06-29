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
        self.prev_str = self.str
        self.prev_thr = self.thr

        self.state = dict()
        self.state['angles']    = [None, None, None]
        self.state['midpoints'] = [None, None, None]
        self.state['x_accel']   = None
        self.state['y_accel']   = None
        self.state['z_accel']   = None
        self.state['prev_angles'] = self.state['angles']

        self.angle_cnt_max = 50
        self.angle_list = [0]

        self.current_factor = 1
        self.history_factor = 0

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

        # Iterating across the lines
        for i in range(3):

            # If the lane is a false value or outlier then you ignore it
            if self.state['prev_angles'][i] is not None:
                if abs(self.state['angles'][i] - self.state['prev_angles'][i]) > 110:
                    self.state['angles'][i] = None

            # Getting average
            if self.state['angles'][i] is None:
                none[i] = True
            else:
                avg_angle += self.state['angles'][i]
                not_none += 1

        # If valid then remember previous value
        if not_none == 0:
            scaled_angle_current = self.prev_str
        # Computing average
        else:
            avg_angle /= not_none

            # print(avg_angle)
            offset_factor = 1
            avg_angle *= offset_factor

            scaled_angle_current = ( (avg_angle/90) * 45 ) + 55

        scaled_angle_history_avg = sum(self.angle_list)/len(self.angle_list)

        if len(self.angle_list) > self.angle_cnt_max:
            del self.angle_list[0]
        self.angle_list.append(scaled_angle_current)

        if 45 < scaled_angle_current < 65:
            self.current_factor = 0.8
            self.history_factor = 0.2
        else:
            self.current_factor = 1
            self.history_factor = 0
        scaled_angle = self.current_factor*scaled_angle_current+self.history_factor*scaled_angle_history_avg

        # Detecting which side of steer
        if 45 < scaled_angle < 65:
            if self.state['angles'][0] is None:
                scaled_angle += 5
            if self.state['angles'][2] is None:
                scaled_angle -= 5


        # self.thr = self.controller.thr

        # Updating steering
        self.str = scaled_angle
        self.prev_str = self.str

        # Updating angles for next state
        self.state['prev_angles'] = self.state['angles']

        # If straightaway then speed up
        if 50 < scaled_angle < 60:
            if 60 <= self.prev_thr <= 80:
                self.thr += 5
            else: # Start at this value
                self.thr = 60
        else:
            # Throttle based on turns
            self.thr = 60 - (abs(avg_angle/90))*(60-45)
            # self.thr = 52




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





