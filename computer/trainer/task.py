import tensorflow as tf
import numpy as np
import time
from os import listdir
from model import Model
import argparse


def train(args):
    model = Model(args.job_dir, args.train_dir, args.batch_size)

    with tf.Session() as sess:

        model.restore(sess)

        # model.eval(sess)

        lowest_loss = model.progress(sess)

        print("Starting training")
        s = time.time()
        try:
            for i in range(args.num_epochs):
                model.batch_update(sess)
                if i % args.save_freq == 0 and i != 0:
                    cur_loss = model.progress(sess)
                    print("step {} - sq error: {}".format(i, cur_loss), time.time()-s)
                    if cur_loss < lowest_loss:
                        model.saveM(sess)
                        lowest_loss = cur_loss
                    else:
                        print("Did not save")
        except KeyboardInterrupt:
            model.clean(sess)
        model.clean(sess)
        print(time.time() -s)


# def test():
#     saver = tf.train.Saver()
#     with tf.Session() as sess:
#         print('Gathering Data...')
#         data = []
#         labels = []
#         s = time.time()
#         for n in range(1, len(listdir('./data'))):
#             print('./data/data{}.npy'.format(n))
#             d = np.load('./data/data{}.npy'.format(n))[0]
#             for i in d:
#                 data.append(i[0])
#
#             l = np.load('./data/data{}.npy'.format(n))[1]
#             for i in l:
#                 labels.append(i[0])
#
#         print(time.time() - s)
#
#         data = np.array(data).reshape(-1, 128, 96, 3)
#         labels = np.array(labels).reshape(-1, 2)
#
#         print(labels.shape, data.shape)
#
#         try:
#             saver.restore(sess, "models/model.ckpt")
#             print('restored')
#         except Exception as e:
#             print('could not load')
#             return
#         for i in range(len(data)):
#             y_o = sess.run([y_out], feed_dict={x_in: data[i].reshape(1, 128, 96, 3)})
#             print("iteration: {}     exp_y: {}     y_out: {}".format(i, labels[i].reshape(1, 2)[0], y_o[0][0]))


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        '--train-dir',
        help='GCS file or local paths to training data',
        )
    PARSER.add_argument(
        '--eval-dir',
        help='GCS file or local paths to eval data',
    )
    PARSER.add_argument(
        '--job-dir',
        help='GCS location to write checkpoints and export models'
        )
    PARSER.add_argument(
        '--num-epochs',
        help='Number of training steps to run',
        type=int)
    PARSER.add_argument(
        '--batch-size',
        help='Batch size for training steps',
        type=int,
        default=200)
    PARSER.add_argument(
        '--save-freq',
        help='epochs between saves',
        type=int,
        default=100)
    PARSER.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')

    ARGUMENTS, _ = PARSER.parse_known_args()
    tf.logging.set_verbosity(ARGUMENTS.verbosity)

    train(ARGUMENTS)

