import tensorflow as tf
import numpy as np
import time
from os import listdir

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=.5)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, s=[1, 2, 2, 1]):
    return tf.nn.conv2d(x, W, strides=s, padding='SAME')


with tf.variable_scope('Supervised'):
    x_in = tf.placeholder(tf.float32, shape=[None, 128, 96, 3])
    exp_y = tf.placeholder(tf.float32, shape=[None, 2])

    W_conv1 = weight_variable([4, 4, 3, 16])
    b_conv1 = bias_variable([16])
    h_conv1 = tf.nn.relu(conv2d(x_in, W_conv1) + b_conv1)

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
    y_out = tf.nn.tanh(tf.matmul(h_fc2, W_fc3) + b_fc3)


    huber = tf.losses.huber_loss(exp_y, y_out)
    train_step = tf.train.AdamOptimizer(5e-5).minimize(huber)
    accuracy = tf.losses.mean_squared_error(exp_y, y_out)

def train():
    saver = tf.train.Saver()
    with tf.Session() as sess:
        print('Gathering Data...')
        data = []
        labels = []
        s = time.time()
        for n in range(1, len(listdir('./data'))):
            print('./data/data{}.npy'.format(n))
            d = np.load('./data/data{}.npy'.format(n))[0]
            for i in d:
                data.append(i[0])

            l = np.load('./data/data{}.npy'.format(n))[1]
            for i in l:
                labels.append(i[0])

        print(time.time() - s)

        data = np.array(data).reshape(-1, 128, 96, 3)
        labels = np.array(labels).reshape(-1, 2)

        print(labels.shape, data.shape)


        try:
            saver.restore(sess, "models/model.ckpt")
            print('restored')
        except:
            print('did not restore')
            sess.run(tf.global_variables_initializer())

        s = time.time()
        ce = 1.0
        i = 0
        sq1 = sess.run([accuracy], feed_dict={x_in: data, exp_y: labels})
        try:
            while(1):
                if i % 300 == 0:

                    acc, sc = sess.run([accuracy, train_step], feed_dict={x_in: data, exp_y: labels})
                    print('step {} - sq error: {}    percent: {}    time: {}'.format(i, acc, 1 - (acc/sq1), time.time() - s))
                    saver.save(sess, "models/model.ckpt")

                # if i % 10 == 0:
                #     print sess.run(output, feed_dict={x_in: data, exp_y: labels, m: len(data)})
                else:
                    sess.run(train_step, feed_dict={x_in: data, exp_y: labels})
                i += 1
                if i == 3000:
                    writer = tf.summary.FileWriter("/tmp/model/1")
                    writer.add_graph(sess.graph)
                    saver.save(sess, "models/model.ckpt")
                    print('saved to models/model.ckpt')
                    tf.train.write_graph(sess.graph.as_graph_def(), '.', 'models/model.pbtxt', as_text=True)
                    print(time.time() - s)
        finally:
            writer = tf.summary.FileWriter("/tmp/model/1")
            writer.add_graph(sess.graph)
            saver.save(sess, "models/model.ckpt")
            print('saved to models/model.ckpt')
            tf.train.write_graph(sess.graph.as_graph_def(), '.', 'models/model.pbtxt', as_text=True)
            print(time.time() - s)

def test():
    saver = tf.train.Saver()
    with tf.Session() as sess:
        print('Gathering Data...')
        data = []
        labels = []
        s = time.time()
        for n in range(1, len(listdir('./data'))):
            print('./data/data{}.npy'.format(n))
            d = np.load('./data/data{}.npy'.format(n))[0]
            for i in d:
                data.append(i[0])

            l = np.load('./data/data{}.npy'.format(n))[1]
            for i in l:
                labels.append(i[0])

        print(time.time() - s)

        data = np.array(data).reshape(-1, 128, 96, 3)
        labels = np.array(labels).reshape(-1, 2)

        print(labels.shape, data.shape)

        try:
            saver.restore(sess, "models/model.ckpt")
            print('restored')
        except Exception as e:
            print('could not load')
            return
        for i in range(len(data)):
            y_o = sess.run([y_out], feed_dict={x_in: data[i].reshape(1, 128, 96, 3)})
            print("iteration: {}     exp_y: {}     y_out: {}".format(i, labels[i].reshape(1, 2)[0], y_o[0][0]))

def testOnes():
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, "models/model.ckpt")
        print('restored')
        y_o = sess.run(y_out, feed_dict={x_in: np.ones([1, 128, 96, 3])})
        print(y_o)


train()