"""
Provides vehicle perception of the environment
"""

import time
import logging
import threading
import numpy as np
from autorc.vehicle.vision.retina import Retina

class CortexAdvanced(threading.Thread):

    def __init__(self, update_interval_ms, oculus, corti, controller):

        # Logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            format='%(asctime)s %(module)s %(levelname)s: %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p',
            level=logging.INFO)
        self.logger.setLevel(logging.INFO)

        # External vehicle interfaces
        self.oculus = oculus
        self.retina = Retina()
        self.corti = corti

        # Retina configuration
        self.retina.fil_hsv_l[2] = 180
        self.retina.fil_hsv_u[1] = 100

        # Thread configuration
        self.thread_name = "Cortex"
        threading.Thread.__init__(self, name=self.thread_name)

        # Thread parameters
        self.enabled = False
        self.update_interval_ms = update_interval_ms

        # Observation space
        self.observation_space = dict()
        self.observation_space['left_lane_present'] = None
        self.observation_space['right_lane_present'] = None
        self.observation_space['splitter_present'] = None
        self.observation_space['left_lane_present'] = None
        self.observation_space['offroad'] = None
        self.observation_space['vehicle_lane_angle'] = None

        self.observation_space['x_acceleration'] = None
        self.observation_space['y_acceleration'] = None
        self.observation_space['z_acceleration'] = None

        self.observation_space['user_throttle'] = None
        self.observation_space['user_steering'] = None

        # Reward
        self.reward = 0

        # Training mode
        self.mode = "IMITATION"

    def get_state(self):

        # Setting the current frame
        self.retina.frame = self.oculus.get_frame()

        # Detecting lines
        if self.retina.frame is not None:
            self.angles , self.midpoints = self.retina.process()

    def compute_reward(self, mode):

        if self.mode == "IMITATION":

            reward = (self.user_thr-self.cerebellum_thr)*(self.user_str - self.cerebellum_str)

        if self.mode == "REINFORCEMENT":

            reward = (1-acceleration)*(1-)

    def enable(self):

        self.enabled = True

    def disable(self):

        self.enabled = False

    def set_mode(self, mode):

        self.mode = mode # Imitation or Reinforcement

    def gaussian_function(self, x, mu):

        """
        :param sigma: std
        :param mu: mean
        :return: value of gaussian function at x
        """

        amplitude = 0.5
        sigma = 1

        return np.exp(-1*amplitude * (((x - mu) / sigma) ** 2))

    def run(self):

        while True:

            if self.enabled == True:

                self.get_state()



            time.sleep(self.update_interval_ms / 1000)

# ------------------------------------------------------------------------------
#                                 SAMPLE CODE
# ------------------------------------------------------------------------------

if __name__ == '__main__':

    retina = Retina()

    retina.load_npy(file_name='/Users/arnavgupta/car_data/raw_npy/oculus-2019-06-16 20;49;28.264824.npy')
    retina.test_line_detection()





