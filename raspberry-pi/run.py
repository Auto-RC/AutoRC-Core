import tensorflow as tf
import numpy as np
import time
from controller import Controller
from carControl import Car_Control
from camera import PiCamera
from threading import Thread


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
    y_out = tf.nn.sigmoid(tf.matmul(h_fc2, W_fc3) + b_fc3)


def calcThrottle(throttle, brake):
    if brake > -0.8:
        return -(brake+1)/2
    else:
        return (throttle+1)/2

controller = Controller()
controller.on = False
drive = Car_Control(60)
cam = PiCamera()

t = Thread(target=cam.update, args=())
t.daemon = True
t.start()
print("started thread")
print('waiting')

with tf.Session() as sess:

    while not controller.on:
        time.sleep(0.1)

    while controller.on:
        frame = np.array(cam.run_threaded()).reshape(1, 128, 96, 3)
        out = sess.run([y_out], feed_dict={x_in: frame})
        drive.set_steering(out[0])
        drive.set_throttle(out[1])