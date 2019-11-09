"""
Deep reinforcement learning module to learn racing
"""

import random
import numpy as np
import logging
import threading
from collections import deque
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from pathlib import Path
import tensorflow as tf
import time
import os
import copy


class CerebellumSupervisedLearning(threading.Thread):
    """
    Cerebellum runs a supervised learning learning neural network
    """

    MODEL_DIR = os.path.join(str(Path.home()), "Github", "AutoRC-Core", "autorc", "models")

    MEMORY_SIZE = 1000000

    BATCH_SIZE = 20

    OBSERVATION_SPACE = 15

    STR_ACTIONS = [-45 / 45, -21 / 45, -9 / 45, -3 / 45, 0, 3 / 45, 9 / 45, 21 / 45, 45 / 45]
    THR_ACTIONS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    # [0 0 0 0 1 0 ] -> User input -> corresponds to [0, -1] # [Throttle, Steering]
    # [0.3 1 0.4 0.5 0.1 0.25 ] -> [0 1 0 0 0 0 ] -> NN output -> corresponds to [0.2, -0.25] # [Throttle, Steering]

    ACTION_SPACE = 2  # len(STR_ACTIONS) * len(THR_ACTIONS)

    GLOBAL_STEP = tf.Variable(0, trainable=False)

    LEARNING_RATE = tf.train.exponential_decay(1e-2,
                                               global_step=GLOBAL_STEP,
                                               decay_steps=5000, decay_rate=0.8)

    ADD_GLOBAL = GLOBAL_STEP.assign_add(1)

    # Turns off dropout if not TRAINING_MODE
    TRAINING_MODE = False

    def __init__(self, update_interval_ms, controller, cortex, corti, model_name, imitation=True, load=True,
                 save=False):

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

        self.init_action_space()

        # External vehicle interfaces
        self.controller = controller
        self.cortex = cortex
        self.corti = corti

        # The number of episodes to store
        self.memory = deque(maxlen=self.MEMORY_SIZE)

        # Model Training Config
        self.save = save

        # Model config
        self.model_name = model_name
        self.save_path = os.path.join(self.MODEL_DIR)
        self.checkpoint = ModelCheckpoint(self.save_path, monitor="loss", verbose=0, save_best_only=False, mode='min')
        self.callbacks_list = [self.checkpoint]

        # The number of batches which have been trained
        self.batches_trained = 0

        # Initialize neural network
        self.config = tf.ConfigProto()

        # Sets GPU Memory Growth Option
        self.config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=self.config)
        self.init_neural_network()
        self.saver = tf.train.Saver()
        if load:
            self.restore()
        else:
            self.sess.run(tf.global_variables_initializer())

        # Initializing parameters
        self.str = 0
        self.thr = 0

    def init_action_space(self):

        self.ACTIONS = dict()
        index = 0
        for thr in self.THR_ACTIONS:
            for str in self.STR_ACTIONS:
                self.ACTIONS[index] = [thr, str]
                index += 1

    def get_action_index(self, action):

        str = action[0][0]
        thr = action[0][1]
        min_thr_error = 100
        for i, throttle_values in enumerate(self.THR_ACTIONS):
            if abs(throttle_values - thr) < min_thr_error:
                thr_index = i
                min_thr_error = abs(throttle_values - thr)

        min_str_error = 100
        for i, str_values in enumerate(self.STR_ACTIONS):
            if abs(str_values - str) < min_str_error:
                str_index = i
                min_str_error = abs(str_values - str)

        index = thr_index * len(self.STR_ACTIONS) + str_index - 1
        print("User Action Index: {}".format(index))
        return index

    def gen_one_hot(self, index):
        vector = np.zeros(self.ACTION_SPACE)
        vector[index] = 1.0
        vector = vector.astype(np.float32)
        return vector

    def get_batches_trained(self):

        return self.batches_trained

    def init_neural_network(self):

        """
        Instantiate the neural network
        """

        # Sets keep_prob of dropout layers given TRAINING_MODE
        keep_prob = 0.7 if self.TRAINING_MODE else 1.0

        # Neural network configuration
        def weight_variable(shape, output_layer=False):
            initial = tf.truncated_normal(shape, stddev=.5)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0, shape=shape)
            return tf.Variable(initial)

        # The input to the network is a 15x1 matrix
        self.x_in = tf.placeholder(tf.float32, shape=[None, self.OBSERVATION_SPACE])
        self.exp_y = tf.placeholder(tf.float32, shape=[None, self.ACTION_SPACE])

        self.h_fc1 = tf.layers.dense(self.x_in, 512, activation=tf.nn.relu,
                                     kernel_initializer=tf.initializers.he_normal())

        # Adding randomness
        self.h_fc1_dropout = tf.nn.dropout(self.h_fc1, keep_prob=keep_prob)

        self.h_fc2 = tf.layers.dense(self.h_fc1_dropout, 512, activation=tf.nn.relu,
                                     kernel_initializer=tf.initializers.he_normal())

        # # Adding randomness
        self.h_fc2_dropout = tf.nn.dropout(self.h_fc2, keep_prob=keep_prob)
        #
        self.h_fc3 = tf.layers.dense(self.h_fc2_dropout, 512, activation=tf.nn.relu,
                                     kernel_initializer=tf.initializers.he_normal())
        #
        # # Adding randomness
        self.h_fc3_dropout = tf.nn.dropout(self.h_fc3, keep_prob=keep_prob)

        # Output of the network is a 99x1 matrix
        # self.y_out = tf.layers.dense(self.h_fc3_dropout, 99, kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.y_out = tf.layers.dense(self.h_fc3_dropout, 2,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.loss = tf.losses.sigmoid_cross_entropy(self.exp_y, self.y_out)
        self.train_step = tf.train.MomentumOptimizer(self.LEARNING_RATE, momentum=0.95).minimize(self.loss)
        # self.sq_error = tf.losses.mean_squared_error(self.exp_y, self.y_out)
        self.graph = tf.get_default_graph()

    def predict(self, x_in):
        print('Input:', x_in)

        output = self.y_out
        network_out = self.sess.run(tf.nn.sigmoid(output), feed_dict={self.x_in: x_in})

        # 0: Steering [-1, 1] -> [0, 1]
        # 1: Throttle [0, 1]
        print('Prediction: ', network_out)

        # Converting the one hot encoding back to a throttle and steering value
        # print("MACHINE ACTION: {}".format(np.argmax(network_out)))
        # print("ONE HOT OUT: {}".format(network_out))
        # print('Machine Action: {}'.format(self.ACTIONS[np.argmax(network_out)]))
        # return self.ACTIONS[np.argmax(network_out)]

        return network_out

    def fit(self, x_in, exp_y):

        # Clips outliers in data to [-1, 1] to prevent gradient explosion
        x_in = np.clip(x_in, -1.0, 1.0)

        exp_y = np.split(exp_y, 2, axis=-1)
        exp_y[0] = (exp_y[0] + 1.0) / 2.0
        exp_y = np.concatenate(exp_y, axis=-1)

        print('Label: ', exp_y)

        # Encoding into one hot vector
        # action_ind = self.get_action_index(exp_y)
        # action_one_hot = self.gen_one_hot(action_ind)
        # action_one_hot = np.reshape(action_one_hot, [1, 99])
        # print("MACHINE ACTION ONE HOT: {}".format(action_one_hot))
        # print('Label:', action_one_hot)

        # Normalize y using tanh
        # exp_y_normalized = np.tanh(exp_y)

        loss, _, _, learning_rate = self.sess.run([self.loss, self.train_step, self.ADD_GLOBAL, self.LEARNING_RATE],
                                                  feed_dict={self.x_in: x_in, self.exp_y: exp_y})
        print('Learning Rate: {}'.format(learning_rate))
        return loss

    def restore(self):
        try:
            self.saver.restore(self.sess, os.path.join(self.save_path, "{}.ckpt".format(self.model_name)))
            print('Restored from', os.path.join(self.save_path, "{}.ckpt".format(self.model_name)))
        except:
            print('Could not restore, randomly initializing all variables')
            self.sess.run(tf.global_variables_initializer())

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
        action = self.predict(state)

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

        # Iterating through the batch
        for state, user_action, terminal_state in batch:

            # Output of the neural network (q values) given the state
            state = np.reshape(state, (1, 15))
            user_action = np.reshape(user_action, (1, 2))

            loss.append(self.fit(state, user_action))

            # Training the model on the updated q_values
            if self.save and terminal_state == 1:
                model_save_path = os.path.join(self.save_path, "{}.ckpt".format(self.model_name))

                # if os.path.exists(model_save_path):
                self.saver.save(self.sess, model_save_path)
                # else:
                #     self.saver.save(self.sess, model_save_path)

            self.batches_trained += 1

            # Updates learning rate

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
