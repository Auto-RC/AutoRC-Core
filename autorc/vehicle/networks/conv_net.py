import tensorflow as tf


class ConvNet:

    def __init__(self, **kwargs):

        # The input to the network is a 15x1 matrix
        self.x_in = tf.placeholder(tf.float32, shape=[None, 33, 128, 3])
        self.exp_y = tf.placeholder(tf.float32, shape=[None, kwargs['action_space']])

        self.n_channels = 8

        self.conv1_1 = tf.layers.conv2d(self.x_in, self.n_channels, (3, 3), strides=(1, 1), padding='same',
                                        activation=tf.nn.relu,
                                        kernel_initializer=tf.initializers.he_normal())

        self.conv1_2 = tf.layers.conv2d(self.conv1_1, self.n_channels, (3, 3),  strides=(1, 1), padding='same',
                                        activation=tf.nn.relu,
                                        kernel_initializer=tf.initializers.he_normal())

        # 16 x 64
        self.max_pool1 = tf.layers.max_pooling2d(self.conv1_2, (2, 2), strides=(2, 2))

        self.conv2_1 = tf.layers.conv2d(self.max_pool1, self.n_channels * 2, (3, 3), strides=(1, 1), padding='same',
                                        activation=tf.nn.relu,
                                        kernel_initializer=tf.initializers.he_normal())

        self.conv2_2 = tf.layers.conv2d(self.conv2_1, self.n_channels * 2, (3, 3),  strides=(1, 1), padding='same',
                                        activation=tf.nn.relu,
                                        kernel_initializer=tf.initializers.he_normal())

        # 8 x 32
        self.max_pool2 = tf.layers.max_pooling2d(self.conv2_2, (2, 2), strides=(2, 2))

        self.conv3_1 = tf.layers.conv2d(self.max_pool2, self.n_channels * 4,  (3, 3),  strides=(1, 1), padding='same',
                                        activation=tf.nn.relu,
                                        kernel_initializer=tf.initializers.he_normal())

        self.conv3_2 = tf.layers.conv2d(self.conv3_1, self.n_channels * 4, (3, 3), strides=(1, 1), padding='same',
                                        activation=tf.nn.relu,
                                        kernel_initializer=tf.initializers.he_normal())

        # 4 x 16
        self.max_pool3 = tf.layers.max_pooling2d(self.conv3_2, (2, 2), strides=(2, 2))

        self.fc_in = tf.layers.flatten(self.max_pool3)
        self.fc1 = tf.layers.dense(self.fc_in, 256, activation=tf.nn.relu,
                                   kernel_initializer=tf.initializers.he_normal())
        self.fc1_dropout = tf.nn.dropout(self.fc1, keep_prob=kwargs['keep_prob'])
        self.fc2 = tf.layers.dense(self.fc1, 256, activation=tf.nn.relu,
                                   kernel_initializer=tf.initializers.he_normal())
        self.fc2_dropout = tf.nn.dropout(self.fc2, keep_prob=kwargs['keep_prob'])
        self.fc3 = tf.layers.dense(self.fc2, 2, activation=tf.nn.sigmoid,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer())
        self.y_out = self.fc3

        self.loss = tf.losses.sigmoid_cross_entropy(self.exp_y, self.y_out)
        self.train_step = tf.train.MomentumOptimizer(learning_rate=kwargs['learning_rate'], momentum=0.95).minimize(self.loss)
        # self.sq_error = tf.losses.mean_squared_error(self.exp_y, self.y_out)
        self.graph = tf.get_default_graph()
