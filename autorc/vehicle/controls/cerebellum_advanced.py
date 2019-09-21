"""
Deep reinforcement learning module to learn racing
"""

import random
import numpy as np
import logging
import threading
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from pathlib import Path
import tensorflow as tf
import time
import os

class CerebellumAdvanced(threading.Thread):

    """
    Cerebellum runs a deep reinforcement learning neural network
    """

    MODEL_DIR = os.path.join(str(Path.home()), "git", "auto-rc_poc", "autorc", "model")

    GAMMA = 1

    EXPLORATION_RATE = 0.2
    EXPLORATION_DECAY = 0.9
    EXPLORATION_MIN = 0.1

    MEMORY_SIZE = 1000000

    BATCH_SIZE = 10

    STR_ACTIONS = [-45/45, -21/45, -9/45, -3/45, 0, 3/45, 9/45, 21/45, 45/45]
    THR_ACTIONS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    OBSERVATION_SPACE = [0 for i in range(0, 15)]

    LEARNING_RATE = 0.001

    def __init__(self, update_interval_ms, controller, cortex, corti, model_name, imitation=True, load=True, train=False):

        """
        Constructor

        :param update_interval_ms: Thread execution period
        :param controller: Interface to user rf controller module
        :param cortex: Environment interface
        :param corti: Interface to inertial measurement systems
        """

        # Logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            format='%(asctime)s %(module)s %(levelname)s: %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p',
            level=logging.INFO)
        self.logger.setLevel(logging.INFO)

        # Thread parameters
        self.thread_name = "Cerebellum"
        threading.Thread.__init__(self, name=self.thread_name)

        # How often to run each step
        self.update_interval_ms = update_interval_ms

        # Default is no auto mode
        self.auto = False

        # External vehicle interfaces
        self.controller = controller
        self.cortex = cortex
        self.corti = corti

        # The number of episodes to store
        self.memory = deque(maxlen=self.MEMORY_SIZE)

        # Model Training Config
        self.imitation = imitation
        self.train = train

        # Model config
        self.model_name = model_name
        self.load = load
        self.save_path = os.path.join(self.MODEL_DIR, self.model_name)
        self.checkpoint = ModelCheckpoint(self.save_path, monitor="loss", verbose=0, save_best_only=False, mode='min')
        self.callbacks_list = [self.checkpoint]

        # Initializing empty state
        self.state = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

        # Initialize action space
        self.init_action_space()

        # Initialize neural network
        self.init_neural_network()

    def init_action_space(self):

        self.ACTION_SPACE = [0 for i in range(0, len(self.STR_ACTIONS) * len(self.THR_ACTIONS))]
        self.ACTIONS = dict()
        index = 0
        for thr in self.THR_ACTIONS:
            for str in self.STR_ACTIONS:
                self.ACTIONS[index] = [thr, str]
                index += 1

    def init_neural_network(self):

        """
        Instantiate the neural network
        """

        # Neural network configuration
        if self.load == False:
            self.model = Sequential()
            self.model.add(Dense(24, input_shape=(len(self.OBSERVATION_SPACE),), activation="relu"))
            self.model.add(Dense(24, activation="relu"))
            self.model.add(Dense(len(self.ACTION_SPACE), activation="linear"))
            self.model.compile(loss="mse", optimizer=Adam(lr=self.LEARNING_RATE))
        else:
            self.model = load_model(self.save_path)

        self.graph = tf.get_default_graph()

    def remember(self, state, action, reward, next_state, offroad):

        """
        Stores the current step

        :param state: Environment state
        :param action: The action taken in this step
        :param reward: The reward for this action
        :param next_state: The next environment state
        :param offroad: Vehicle offroad flag
        """

        self.memory.append((state, action, reward, next_state, offroad))

    def act(self, state):

        """
        Takes an action given the environment state

        :param state: Environment state
        """

        # If randomness is below threshold choose a random action
        if np.random.rand() < self.EXPLORATION_RATE:
            return random.randrange(len(self.ACTION_SPACE))
        # Otherwise choose an action based on the neural network
        else:
            with self.graph.as_default():
                q_values = self.model.predict(state)

            return np.argmax(q_values[0])

    def experience_replay(self):

        """
        Train the model based on stored experience
        """

        # If there are not enough steps in the episode
        # then we cannot sample a full batch
        if len(self.memory) < self.BATCH_SIZE:
            return

        # Sample a random batch from memory
        batch = random.sample(self.memory, self.BATCH_SIZE)

        # Iterating through the batch
        for state, action, reward, state_next, terminal_state in batch:

            # Initial reward
            q_update = reward

            # If the episode is not yet failed
            if not terminal_state:

                # Bellman equation
                # TODO: Why is there a zero index [0]?
                state_next = np.reshape(state_next, (1, 15))
                q_update = (reward + self.GAMMA*np.amax(self.model.predict(state_next)[0]))

            # Output of the neural network (q values) given the state
            state = np.reshape(state, (1, 15))
            q_values = self.model.predict(state)

            # Action is the action which was taken in the state during the
            # actual episode. This action is/was thought to be the optimal
            # action before training. This action gets updated with the new
            # reward.
            q_values[0][action] = q_update

            # Training the model on the updated q_values
            if self.train:
                self.model.fit(state, q_values, verbose=0, callbacks=self.callbacks_list)
            else:
                self.model.fit(state, q_values, verbose=0)

        # Updating the exploration rate
        self.exploration_rate *= self.EXPLORATION_DECAY

        # Capping the exploration rate
        self.exploration_rate = max(self.EXPLORATION_MIN, self.exploration_rate)

    def update_state(self, state):

        self.state = np.array([state])

    def compute_controls(self):

        action_index = self.act(self.state)
        return {
            "action": self.ACTIONS[action_index],
            "index" : action_index
        }

    def run(self):

        while True:

            if self.auto == False:
                self.thr = self.controller.thr
                self.str = self.controller.str
            elif self.auto == True:
                self.thr, self.str = self.compute_controls()

            time.sleep(self.update_interval_ms / 1000)