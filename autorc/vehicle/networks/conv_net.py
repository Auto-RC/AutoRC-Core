import tensorflow as tf

class ConvNet:

    def __init__(self, **kwargs):

        # Neural network configuration
        def weight_variable(shape, output_layer=False):
            initial = tf.truncated_normal(shape, stddev=.5)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0, shape=shape)
            return tf.Variable(initial)

        # The input to the network is a 15x1 matrix
        self.x_in = tf.placeholder(tf.float32, shape=[None, kwargs['observation_space'])
        self.exp_y = tf.placeholder(tf.float32, shape=[None, kwargs['action_space'])

        self.h_fc1 = tf.layers.dense(self.x_in, 512, activation=tf.nn.relu,
                                     kernel_initializer=tf.initializers.he_normal())

        # Adding randomness
        self.h_fc1_dropout = tf.nn.dropout(self.h_fc1, keep_prob=kwargs['keep_prob'])

        self.h_fc2 = tf.layers.dense(self.h_fc1_dropout, 512, activation=tf.nn.relu,
                                     kernel_initializer=tf.initializers.he_normal())

        # # Adding randomness
        self.h_fc2_dropout = tf.nn.dropout(self.h_fc2, keep_prob=kwargs['keep_prob'])
        #
        self.h_fc3 = tf.layers.dense(self.h_fc2_dropout, 512, activation=tf.nn.relu,
                                     kernel_initializer=tf.initializers.he_normal())
        #
        # # Adding randomness
        self.h_fc3_dropout = tf.nn.dropout(self.h_fc3, keep_prob=kwargs['keep_prob'])

        # Output of the network is a 99x1 matrix
        # self.y_out = tf.layers.dense(self.h_fc3_dropout, 99, kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.y_out = tf.layers.dense(self.h_fc3_dropout, 2,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer())

        self.loss = tf.losses.sigmoid_cross_entropy(self.exp_y, self.y_out)
        self.train_step = tf.train.MomentumOptimizer(self.LEARNING_RATE, momentum=0.95).minimize(self.loss)
        # self.sq_error = tf.losses.mean_squared_error(self.exp_y, self.y_out)
        self.graph = tf.get_default_graph()
