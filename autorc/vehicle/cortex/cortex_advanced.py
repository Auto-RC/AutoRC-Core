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

    CONTROL_OPTIMIZATION = True

    def __init__(self, update_interval_ms, oculus, corti, drive, cerebellum, mode="simulation"):

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
        self.cerebellum = cerebellum

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
        self.observation_space['left_lane_present'] = 1
        self.observation_space['right_lane_present'] = 1
        self.observation_space['splitter_present'] = 1
        self.observation_space['vehicle_offroad'] = 0

        # Position Observations
        self.observation_space['left_lane_position'] = -50
        self.observation_space['right_lane_position'] = 50
        self.observation_space['splitter_position'] = 0
        self.observation_space['vehicle_position'] = 0

        # Angle observations
        self.observation_space['left_lane_angle'] = 70
        self.observation_space['right_lane_angle'] = -70
        self.observation_space['splitter_angle'] = 0
        self.observation_space['vehicle_angle'] = 0

        # Observation space acceleration
        self.observation_space['x_acceleration'] = 0
        self.observation_space['y_acceleration'] = 0
        self.observation_space['z_acceleration'] = 0

        # Observation space user controls
        self.observation_space['user_throttle'] = 0
        self.observation_space['user_steering'] = 0.5
        self.observation_space['terminal'] = 0

        # Initializing previous observation state
        self.prev_observation_space = self.observation_space

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

            self.prev_observation_space = self.observation_space

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
            raise e


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

    def compute_controls(self):

        action = self.cerebellum.compute_controls()[0]
        raw_str = action[0]
        raw_thr = action[1]

        print("raw_thr: {} raw_str: {}".format(raw_thr, raw_str))

        self.thr = raw_thr
        self.str = raw_str

        try:

            if (self.CONTROL_OPTIMIZATION) and (self.check_retina_confidence() == 1):

                    self.thr = self.optimize_throttle(raw_thr, raw_str)
                    self.str = self.correct_steering(raw_str)

                    print("Updated throttle from {} to {}".format(raw_thr, self.thr))
                    print("Updated steering from {} to {}".format(raw_str, self.str))

        except Exception as e:
            print(e)
            pass

        # Bounding Throttle
        self.thr = min(self.thr, 1)

        # Bounding steering
        if self.str > 1:
            self.str = 1
        elif self.str < -1:
            self.str = -1

        print("thr: {} str: {}".format(self.thr, self.str))

        return [self.str, self.thr]

    def check_retina_confidence(self):

        """
        1) If splitter_angle is more than 80deg or less than -80def then low confidence
        2) If vehicle angle > 0.8 and vehicle angle < 0.8 the low confidence
        3) If abs(splitter_angle - prev_splitter_angle) > 40deg then low confidence
        4) If abs(vehicle_angle - prev_vehicle_angle) > 0.3 then low confidence
        """

        splitter_angle_diff = abs(self.observation_space['splitter_angle'] - self.prev_observation_space['splitter_angle'])
        vehicle_position_diff = abs(self.observation_space['vehicle_position'] - self.prev_observation_space['vehicle_position'])

        if (self.observation_space['splitter_angle'] < -80) or (self.observation_space['splitter_angle'] > 80):
            self.confidence = 0
            print("Confidence is zero based on splitter angle")
        elif (self.observation_space['vehicle_position'] < -1) or (self.observation_space['vehicle_position'] > 1):
            self.confidence = 0
            print("Confidence is zero based on vehicle position")
        elif (splitter_angle_diff > 40):
            self.confidence = 0
            print("Confidence is zero based on splitter diff")
        elif (vehicle_position_diff > 0.3):
            self.confidence = 0
            print("Confidence is zero based on vehicle position diff")
        else:
            self.confidence = 1

        print("Retina confidence is {}".format(self.confidence))

        return self.confidence

    def optimize_throttle(self, throttle, steering):

        """
        Checking if there is an opportuinity for the vehicle to increase
        throttle based on splitter angle and vehicle position
        """

        if (abs(steering) > 0.5) and (throttle > 0.3):
            throttle = 0.3

        elif (abs(steering) < 0.5):

            if (self.observation_space['splitter_angle'] < 10) and (self.observation_space['splitter_angle'] > -10):
                throttle += 0.2

                if (self.observation_space['vehicle_position'] < 0.2) or (self.observation_space['vehicle_angle'] > -0.2):
                    throttle += 0.1

            elif (self.observation_space['vehicle_position'] < 0.2) or (self.observation_space['vehicle_angle'] > -0.2):
                throttle += 0.1

        return throttle

    def correct_steering(self, steering):

        if (self.observation_space['vehicle_position'] < 0) and (self.observation_space['splitter_angle'] > 25):
            if (steering < 0):
                steering += 0.3
            elif (steering > 0):
                steering += 0.1

        elif (self.observation_space['vehicle_position'] > 0) and (self.observation_space['splitter_angle'] < -25):
            if (steering > 0):
                steering -= 0.3
            elif (steering < 0):
                steering -= 0.1

        # elif (self.observation_space['vehicle_position'] < -0.25) and (self.observation_space['splitter_angle'] < -50):
        #     steering -= 0.05
        #
        # elif (self.observation_space['vehicle_position'] > 0.25) and (self.observation_space['splitter_angle'] > 50):
        #     steering += 0.05

        elif (self.observation_space['vehicle_position'] > 0.1):
            steering -= 0.1

        elif (self.observation_space['vehicle_position'] < -0.1):
            steering += 0.1

        return steering

    def run(self):

        """
        Cortex thread
        """

        #TODO: Add time tracking (needed for terminal)

        while True:

            if self.enabled == True:

                self.get_state()

            time.sleep(self.update_interval_ms / 1000)
