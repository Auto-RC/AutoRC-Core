import tensorflow as tf
import numpy as np
import os
from random import randint

class Model:
    def __init__(self, save_dir, data_dir, batch_size):
        self.x_in = tf.placeholder(tf.float32, shape=[None, 128, 96, 3])
        self.exp_y = tf.placeholder(tf.float32, shape=[None, 2])
        self.learning_rate = 1e-5
        self.eval_x = tf.placeholder(tf.float32, shape=[None, 128, 96, 3])
        self.eval_y = tf.placeholder(tf.float32, shape=[None, 2])
        self.build_model()
        self.save_dir = save_dir

        self.saver = tf.train.Saver()
        self.batch_size = batch_size
        self.load_data(data_dir)
        if self.batch_size > self.data_len:
            self.batch_size = int(self.data_len/5)
            print("Batch size is greater than the data length")
            print("Batch size changed to", self.batch_size)

    def load_data(self, directory):
        print('Gathering Data...')
        data = []
        labels = []
        for filename in os.listdir(directory):
            if filename.endswith(".npy"):
                print("Found data from", os.path.join(directory, filename))
                rdata = np.load(os.path.join(directory, filename))

                d = rdata[0]
                for i in d:
                    data.append(i[0])

                l = rdata[1]
                for i in l:
                    labels.append(i[0])

        self.data = np.array(data).reshape(-1, 128, 96, 3)
        self.labels = np.array(labels).reshape(-1, 2)
        self.data_len = len(data)
        print("Data size:", self.data_len)

    def restore(self,sess):
        try:
            self.saver.restore(sess, os.path.join(self.save_dir, "model.ckpt"))
            print('Restored from', os.path.join(self.save_dir, "model.ckpt"))
        except:
            print('Could not restore, randomly initializing all variables')
            sess.run(tf.global_variables_initializer())

    def build_model(self):
        def weight_variable(shape):
            initial = tf.truncated_normal(shape, stddev=.5)
            return tf.Variable(initial)

        def bias_variable(shape):
            initial = tf.constant(0.1, shape=shape)
            return tf.Variable(initial)

        def conv2d(x, W, s=[1, 2, 2, 1]):
            return tf.nn.conv2d(x, W, strides=s, padding='SAME')

        with tf.variable_scope('Supervised'):
            W_conv1 = weight_variable([4, 4, 3, 16])
            b_conv1 = bias_variable([16])
            h_conv1 = tf.nn.relu(conv2d(self.x_in, W_conv1) + b_conv1)

            W_conv2 = weight_variable([2, 2, 16, 32])
            b_conv2 = bias_variable([32])
            h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)

            W_conv3 = weight_variable([2, 2, 32, 64])
            b_conv3 = bias_variable([64])
            h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, s=[1, 1, 1, 1]) + b_conv3)

            h_conv3_flat = tf.reshape(h_conv3, [-1, 128 * 96 * 4])

            W_fc1 = weight_variable([128 * 96 * 4, 64])
            b_fc1 = bias_variable([64])
            h_fc1 = tf.nn.sigmoid(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

            W_fc2 = weight_variable([64, 64])
            b_fc2 = bias_variable([64])
            h_fc2 = tf.nn.sigmoid(tf.matmul(h_fc1, W_fc2) + b_fc2)

            W_fc3 = weight_variable([64, 2])
            b_fc3 = bias_variable([2])

            self.y_out = tf.nn.tanh(tf.matmul(h_fc2, W_fc3) + b_fc3)

            self.loss = tf.losses.huber_loss(self.exp_y, self.y_out)
            self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            self.sq_error = tf.losses.mean_squared_error(self.exp_y, self.y_out)

    def forward(self, sess, x_in):
        return sess.run(self.y_out, feed_dict={self.x_in: x_in})


    def batch_update(self, sess):
        batch_data = []
        batch_labels = []
        for i in range(self.batch_size):
            index = randint(0, self.data_len - 1)
            batch_data.append(self.data[index])
            batch_labels.append(self.labels[index])
        sess.run(self.train_step, feed_dict={self.x_in: batch_data, self.exp_y: batch_labels})

    def update(self, sess):
        sess.run(self.train_step, feed_dict={self.x_in: self.data, self.exp_y: self.labels})


    def progress(self, sess):
        return sess.run(self.sq_error, feed_dict={self.x_in: self.data, self.exp_y: self.labels})

    def test_ones(self, sess):
        self.saver.restore(sess, os.path.join(self.save_dir, "model.ckpt"))
        print('restored')
        y_o = sess.run(self.y_out, feed_dict={self.x_in: np.ones([1, 128, 96, 3])})
        print(y_o)

    def eval(self, sess):
        y_out = sess.run(self.y_out, feed_dict={self.x_in: self.data})
        for i in range(self.data_len):
            print("Expected: {}, Ouputted: {}".format(self.labels[i], y_out[i]))

    def saveM(self, sess):
        self.saver.save(sess, os.path.join(self.save_dir, "model.ckpt"))
        print('saved to ' + os.path.join(self.save_dir, "model.ckpt"))

    def clean(self, sess):
        writer = tf.summary.FileWriter("/tmp/model/1")
        writer.add_graph(sess.graph)
        self.saver.save(sess, os.path.join(self.save_dir, "model.ckpt"))
        print('saved to ' + os.path.join(self.save_dir, "model.ckpt"))
        tf.train.write_graph(sess.graph.as_graph_def(), '.', os.path.join(self.save_dir, 'model.pbtxt'), as_text=True)
