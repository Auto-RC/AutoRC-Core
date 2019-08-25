"""
Vehicle perception
"""

__author__ = "Anish Agarwal, Arnav Gupta"

import time
import logging
import threading
import numpy as np
from autorc.vehicle.vision.retina import Retina

class CortexAdvanced(threading.Thread):

    """
    Cortex provides perception via vision and inertial systems
    """

    def __init__(self, update_interval_ms, oculus, corti, controller, mode="simulation"):

        """
        Constructor

        :param update_interval_ms: Thread execution period
        :param oculus: Interface to vision systems
        :param corti: Interface to inertial measurement systems
        :param controller: Interface to user rf controller module
        """

        # Logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            format='%(asctime)s %(module)s %(levelname)s: %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p',
            level=logging.INFO)
        self.logger.setLevel(logging.INFO)

        # External vehicle interfaces
        self.retina = Retina()
        self.corti = corti
        self.controller = controller

        self.oculus = oculus

        # Retina configuration
        self.retina.fil_hsv_l[2] = 180
        self.retina.fil_hsv_u[1] = 100

        # Thread configuration
        self.thread_name = "Cortex"
        threading.Thread.__init__(self, name=self.thread_name)

        # Thread parameters
        self.enabled = False
        self.update_interval_ms = update_interval_ms

        # Observation space flags
        self.observation_space = dict()
        self.observation_space['left_lane_present'] = None
        self.observation_space['right_lane_present'] = None
        self.observation_space['splitter_present'] = None
        self.observation_space['left_lane_present'] = None
        self.observation_space['left_lane_position'] = None
        self.observation_space['right_lane_position'] = None
        self.observation_space['splitter_position'] = None
        self.observation_space['offroad'] = None

        # Observation space angles
        self.observation_space['vehicle_lane_angle'] = None

        # Observation space acceleration
        self.observation_space['x_acceleration'] = None
        self.observation_space['y_acceleration'] = None
        self.observation_space['z_acceleration'] = None

        # Observation space user controls
        self.observation_space['user_throttle'] = None
        self.observation_space['user_steering'] = None

        # Reward
        self.reward = 0

        # Training mode
        self.mode = "IMITATION"

        # Offroad State Machine
        self.offroad_sm = []

    def get_state(self):

        """
        Getting the observation space
        """

        # Setting the current frame
        self.retina.frame = self.oculus.get_frame()

        # Detecting lines
        if self.retina.frame is not None:
            self.angles, self.midpoints = self.retina.process()

    def compute_reward(self, cerebellum_thr, cerebellum_str):

        """
        Computes the reward given the training mode, throttle and steering

        :param mode: Imitation vs Reinforcement
        :param cerebellum_thr: Machine computed throttle
        :param cerebellum_str: Machine computed steerting
        :return: Returns the reward
        """

        if self.mode == "IMITATION":
            self.reward = (self.controller.thr - cerebellum_thr)*(self.controller.str - cerebellum_str)

        elif self.mode == "REINFORCEMENT":
            self.reward = 0

    def enable(self):

        """
        Enables the cortex thread
        """

        self.enabled = True

    def disable(self):

        """
        Disables the cortex thread
        """

        self.enabled = False

    def set_mode(self, mode):

        """
        Setting the training mode
        """

        self.mode = mode

    def offroad(self):

        if len(self.offroad_sm) == 5:
            del self.offroad_sm[0]

        self.offroad_sm.append([self.observation_space['splitter_present'], self.observation_space['left_lane_present'], self.observation_space['right_lane_present'], self.observation_space['offroad']])

        offroad = True

        left = False
        right = False

        for i in self.offroad_sm:
            if i[0]:
                offroad = False
            elif i[1]:
                left = True
            elif i[2]:
                right = True

        if offroad:
            if left and right:
                offroad = False
            if left and not right:
                if self.observation_space['left_lane_position'] < 0.5:
                    offroad = False
            if right and not left:
                if self.observation_space['right_lane_position'] > -0.5:
                    offroad = False

        self.observation_space['offroad'] = offroad

    def run(self):

        """
        Cortex thread
        """

        while True:

            if self.enabled == True:

                self.get_state()

            time.sleep(self.update_interval_ms / 1000)
