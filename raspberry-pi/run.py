from controller import Controller
from camera import PiCamera
from carControl import Car_Control
from threading import Thread
import time
import tensorflow as tf
import numpy as np

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

    h_conv3_flat = tf.reshape(h_conv3, [-1, 128 * 96 * 2])

    W_fc1 = weight_variable([128 * 96 * 2, 64])
    b_fc1 = bias_variable([64])
    h_fc1 = tf.nn.sigmoid(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    W_fc2 = weight_variable([64, 64])
    b_fc2 = bias_variable([64])
    h_fc2 = tf.nn.sigmoid(tf.matmul(h_fc1, W_fc2) + b_fc2)

    W_fc3 = weight_variable([64, 2])
    b_fc3 = bias_variable([2])
    y_out = tf.nn.sigmoid(tf.matmul(h_fc2, W_fc3) + b_fc3)

saver = tf.train.Saver()
cam = PiCamera()
# controller = Controller()
drive = Car_Control(60)

t = Thread(target=cam.update, args=())
t.daemon = True
t.start()
print("started camera thread")
time.sleep(1)
print("running")

with tf.Session() as sess:
    try:
        saver.restore(sess, "/home/pi/raspberry-pi/model.ckpt")
    except Exception as e:
        print(e)
        print("Could not load model")
        print("Initializing with random values")
        sess.run(tf.global_variables_initializer())
    while 1:
        t = time.time()
        # if controller.capturing:
        frame = np.array(cam.run_threaded()).reshape([1, 128, 96, 3])
        outs = sess.run(y_out, feed_dict={x_in: frame})
        print(outs, time.time() - t)
