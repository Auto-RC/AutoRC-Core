"""
Deep reinforcement learning module to learn racing logic
"""

import random
import numpy as np
import logging
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class CerebellumAdvanced():

    GAMMA = 1

    EXPLORATION_RATE = 0.2
    EXPLORATION_DECAY = 0.9
    EXPLORATION_MIN = 0.1

    MEMORY_SIZE = 1000000

    BATCH_SIZE = 10

    ACTION_SPACE = [ ( i, [ j for j in range(0,10) ]) for i in range(0,10) ]
    OBSERVATION_SPACE = [0 for 0 in range(0, 12)]

    LEARNING_RATE = 0.001

    def __init__(self, update_interval_ms, controller, cortex, corti):

        # Logger
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            format='%(asctime)s %(module)s %(levelname)s: %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p',
            level=logging.INFO)
        self.logger.setLevel(logging.INFO)

        # How often to run each step
        self.update_interval_ms = update_interval_ms

        # External vehicle interfaces
        self.controller = controller
        self.cortex = cortex
        self.corti = corti

        # The number of episodes to store
        self.memory = deque(maxlen=self.MEMORY_SIZE)

        # Initialize neural network
        self.init_neural_network()

    def init_neural_network(self):

        # Neural network configuration
        self.model = Sequential()
        self.model.add(Dense(24, input_shape=(self.OBSERVATION_SPACE,), activation="relu"))
        self.model.add(Dense(24, activation="relu"))
        self.model.add(Dense(self.ACTION_SPACE, activation="linear"))
        self.model.compile(loss="mse", optimizer=Adam(lr=self.LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):

        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):

        # If randomness is below threshold choose a random action
        if np.random.rand() < self.EXPLORATION_RATE:
            return random.randrange(self.ACTION_SPACE)
        # Otherwise choose an action based on the neural network
        else:
            q_values = self.model.predict(state)
            return np.argmax(q_values[0])

    def experience_replay(self):

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
                q_update = (reward + self.GAMMA*np.amax(self.model.predict(state_next)[0]))

            # Output of the neural network (q values) given the state
            q_values = self.model.predict(state)

            # Action is the action which was taken in the state during the
            # actual episode. This action is/was thought to be the optimal
            # action before training. This action gets updated with the new
            # reward.
            q_values[0][action] = q_update

            # Training the model on the updated q_values
            self.model.fit(state, q_values, verbose=0)

        # Updating the exploration rate
        self.exploration_rate *= self.EXPLORATION_DECAY

        # Capping the exploration rate
        self.exploration_rate = max(self.EXPLORATION_MIN, self.exploration_rate)



