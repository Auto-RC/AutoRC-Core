"""
Vehicle perception
"""

__author__ = "Anish Agarwal, Arnav Gupta"

import time
import logging
import threading
import numpy as np

from autorc.vehicle.vision.retina import Retina
from autorc.vehicle.cortex.environment.lap_history import LapHistory

class CortexAdvanced(threading.Thread):

    """
    Cortex provides perception via vision and inertial systems
    """

    def __init__(self, update_interval_ms, oculus, corti, drive, mode="simulation"):

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
        self.drive = drive
        self.oculus = oculus

        # Lap History
        self.lap_history = LapHistory(memory_size = 5)

        # Thread configuration
        self.thread_name = "Cortex"
        threading.Thread.__init__(self, name=self.thread_name)

        # Thread parameters
        self.enabled = False
        self.update_interval_ms = update_interval_ms

        # Observation feature existence
        self.observation_space = dict()
        self.observation_space['left_lane_present'] = None
        self.observation_space['right_lane_present'] = None
        self.observation_space['splitter_present'] = None
        self.observation_space['vehicle_offroad'] = None

        # Position Observations
        self.observation_space['left_lane_position'] = None
        self.observation_space['right_lane_position'] = None
        self.observation_space['splitter_position'] = None
        self.observation_space['vehicle_position'] = None

        # Angle observations
        self.observation_space['left_lane_angle'] = None
        self.observation_space['right_lane_angle'] = None
        self.observation_space['splitter_angle'] = None
        self.observation_space['vehicle_angle'] = None

        # Observation space acceleration
        self.observation_space['x_acceleration'] = 0
        self.observation_space['y_acceleration'] = 0
        self.observation_space['z_acceleration'] = 0

        # Observation space user controls
        self.observation_space['user_throttle'] = None
        self.observation_space['user_steering'] = None
        self.observation_space['terminal'] = 0

        # Reward
        self.reward = 0

        # Training mode
        self.mode = "IMITATION"

        # Offroad State Machine
        self.offroad_sm = []

        # State counter
        self.state_counter = 0

    def get_state(self):

        """
        Getting the observation space
        """

        # Setting the current frame
        self.retina.frame = self.oculus.get_frame()

        # Making a prediction of what is expected in the next step
        # Retina compares this prediction to its computed value
        if len(self.lap_history.lap) > 0:
            self.retina.prediction = self.lap_history.predict()

        # Detecting lines
        try:

            road = self.retina.process()

            # Adding the current snapshot to the track history
            self.lap_history.add_road_snapshot(road)

            self.observation_space['left_lane_present'] = road.left_lane.present
            self.observation_space['right_lane_present'] = road.right_lane.present
            self.observation_space['splitter_present'] = road.splitter.present
            self.observation_space['vehicle_offroad'] = road.vehicle.offroad

            self.observation_space['left_lane_position'] = road.left_lane.midpoint
            self.observation_space['right_lane_position'] = road.right_lane.midpoint
            self.observation_space['splitter_position'] = road.splitter.midpoint
            self.observation_space['vehicle_position'] = road.vehicle.position

            self.observation_space['left_lane_angle'] = road.left_lane.angle
            self.observation_space['right_lane_angle'] = road.right_lane.angle
            self.observation_space['splitter_angle'] = road.splitter.angle
            self.observation_space['vehicle_angle'] = road.vehicle.angle

            self.observation_space['x_acceleration'] = self.corti.get_frame()
            self.observation_space['y_acceleration'] = 0
            self.observation_space['z_acceleration'] = 0

            self.observation_space['user_throttle'], self.observation_space['user_steering'] = self.drive.get_frame()

            self.state_counter += 1

            if (self.state_counter % 20 == 0) or (self.observation_space['vehicle_offroad']):
                self.observation_space['terminal'] = 1
            else:
                self.observation_space['terminal'] = 0

        except Exception as e:
            print(e)


    def compute_reward(self, cerebellum_thr, cerebellum_str):

        """
        Computes the reward given the training mode, throttle and steering

        :param mode: Imitation vs Reinforcement
        :param cerebellum_thr: Machine computed throttle
        :param cerebellum_str: Machine computed steerting
        :return: Returns the reward
        """

        if self.mode == "IMITATION":


            user_throttle , user_steering = self.drive.get_frame()

            x = user_throttle - cerebellum_thr
            throttle_reward = 5*self.gaussian_function(x, 0.25, 0)

            y = user_steering - cerebellum_str
            steering_reward = 5*self.gaussian_function(y, 0.25, 0)

            self.reward = throttle_reward*steering_reward

        elif self.mode == "REINFORCEMENT":
            self.reward = 0

        if self.reward < 0:
            print("impossible negative reward")

        return self.reward

    def gaussian_function(self, x, sigma, mu):

        """
        Gaussian function used in computing reward

        :param sigma: std
        :param mu: mean
        :return: value of gaussian function at x
        """

        return np.exp(-0.5 * (((x - mu) / sigma) ** 2))

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

    def vectorize_state(self):

        """
        Vectorizing the environment state into a form readable by the neural network
        """

        self.vectorized_state = []

        if self.observation_space['left_lane_present']:
            self.vectorized_state.append(1)
        else:
            self.vectorized_state.append(0)

        if self.observation_space['right_lane_present']:
            self.vectorized_state.append(1)
        else:
            self.vectorized_state.append(0)

        if self.observation_space['splitter_present']:
            self.vectorized_state.append(1)
        else:
            self.vectorized_state.append(0)

        if self.observation_space['vehicle_offroad']:
            self.vectorized_state.append(1)
        else:
            self.vectorized_state.append(0)

        #TODO: How does this -1 affect the relu in the neural network?

        try:
            self.vectorized_state.append(self.observation_space['left_lane_position']/128)
        except:
            self.vectorized_state.append(-1)

        try:
            self.vectorized_state.append(self.observation_space['right_lane_position']/128)
        except:
            self.vectorized_state.append(-1)

        try:
            self.vectorized_state.append(self.observation_space['splitter_position']/128)
        except:
            self.vectorized_state.append(-1)

        try:
            self.vectorized_state.append(self.observation_space['vehicle_position']/128)
        except:
            self.vectorized_state.append(-1)

        try:
            self.vectorized_state.append(self.observation_space['left_lane_angle']/180+0.5)
        except:
            self.vectorized_state.append(-1)

        try:
            self.vectorized_state.append(self.observation_space['right_lane_angle']/180+0.5)
        except:
            self.vectorized_state.append(-1)

        try:
            self.vectorized_state.append(self.observation_space['splitter_angle']/180+0.5)
        except:
            self.vectorized_state.append(-1)

        try:
            self.vectorized_state.append(self.observation_space['vehicle_angle']/180+0.5)
        except:
            self.vectorized_state.append(-1)

        self.vectorized_state.append(self.observation_space['x_acceleration'][0]/10)
        self.vectorized_state.append(self.observation_space['y_acceleration']/10)
        self.vectorized_state.append(self.observation_space['z_acceleration']/10)

        return np.array(self.vectorized_state)

    def get_raw_state(self):

        return self.retina.frame

    def run(self):

        """
        Cortex thread
        """

        #TODO: Add time tracking (needed for terminal)

        while True:

            if self.enabled == True:

                self.get_state()

            time.sleep(self.update_interval_ms / 1000)
