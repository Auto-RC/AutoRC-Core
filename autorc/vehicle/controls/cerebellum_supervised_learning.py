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

class CerebellumSupervisedLearning(threading.Thread):

    """
    Cerebellum runs a supervised learning learning neural network
    """

    MODEL_DIR = os.path.join(str(Path.home()), "git", "AutoRC-Core", "autorc", "models")

    GAMMA = 1

    EXPLORATION_MAX = 0.5
    EXPLORATION_DECAY = 0.999
    EXPLORATION_MIN = 0.3

    MEMORY_SIZE = 1000000

    BATCH_SIZE = 20

    ACTION_SPACE = [0 for i in range(0,2)]

    OBSERVATION_SPACE = [0 for j in range(0, 15)]

    LEARNING_RATE = 0.001

    def __init__(self, update_interval_ms, controller, cortex, corti, model_name, imitation=True, load=True, save=False):

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

        # Initializing empty state
        self.state = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

        # External vehicle interfaces
        self.controller = controller
        self.cortex = cortex
        self.corti = corti

        # The number of episodes to store
        self.memory = deque(maxlen=self.MEMORY_SIZE)

        # Model Training Config
        self.imitation = imitation
        self.save = save

        # Model config
        self.model_name = model_name
        self.load = load
        self.save_path = os.path.join(self.MODEL_DIR, self.model_name)
        self.checkpoint = ModelCheckpoint(self.save_path, monitor="loss", verbose=0, save_best_only=False, mode='min')
        self.callbacks_list = [self.checkpoint]

        # The number of batches which have been trained
        self.batches_trained = 0

        # Initialize neural network
        self.init_neural_network()

        # Initializing parameters
        self.str = 0
        self.thr = 0

    def get_exploration_rate(self):

        return self.exploration_rate

    def get_batches_trained(self):

        return self.batches_trained

    def init_neural_network(self):

        """
        Instantiate the neural network
        """

        # Neural network configuration
        if self.load == False:
            self.model = Sequential()
            self.model.add(Dense(24, input_shape=(len(self.OBSERVATION_SPACE),), activation="sigmoid"))
            self.model.add(Dense(24, activation="sigmoid"))
            self.model.add(Dense(len(self.ACTION_SPACE), activation="linear"))
            self.model.compile(loss="mse", optimizer=Adam(lr=self.LEARNING_RATE))
        else:
            self.model = load_model(self.save_path)

        self.graph = tf.get_default_graph()

    def remember(self, state, user_action, terminal):

        """
        Stores the current step

        :param state: Environment state
        :param action: The action taken in this step
        :param reward: The reward for this action
        :param next_state: The next environment state
        :param offroad: Vehicle offroad flag
        """

        self.memory.append((state, user_action, terminal))

    def act(self, state):

        """
        Takes an action given the environment state

        :param state: Environment state
        """

        # If randomness is below threshold choose a random action
        action = self.model.predict(state)

        return action

    def experience_replay(self):

        """
        Train the model based on stored experience
        """

        # If there are not enough steps in the episode
        # then we cannot sample a full batch
        if len(self.memory) < self.BATCH_SIZE:
            return -1

        # Sample a random batch from memory
        batch = random.sample(self.memory, self.BATCH_SIZE)

        # The loss values across the entire batch
        loss = []
        rewards = []

        # Iterating through the batch
        for state, user_action, terminal_state in batch:

            # Output of the neural network (q values) given the state
            state = np.reshape(state, (1, 15))
            user_action = np.reshape(user_action, (1, 2))

            # Training the model on the updated q_values
            if self.save and terminal_state == 1:
                history = self.model.fit(state, user_action, verbose=0, callbacks=self.callbacks_list)
            else:
                history = self.model.fit(state, user_action, verbose=0)

            loss = loss + history.history['loss']

            self.batches_trained += 1

        # Returning the average loss if loss list is not empty
        return np.mean(loss)

    def update_state(self, state):

        self.state = np.array([state])

    def compute_controls(self):

        return self.act(self.state)

    def run(self):

        time.sleep(2000)

        while True:

            if self.auto == False:
                self.thr = self.controller.thr
                self.str = self.controller.str
            elif self.auto == True:
                self.thr, self.str = self.compute_controls()

            time.sleep(self.update_interval_ms / 1000)