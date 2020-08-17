from __future__ import print_function

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../mnist/", one_hot=False)
X_test, Y_test = mnist.test.images, mnist.test.labels


with tf.Session() as sess:
    saver = tf.train.import_meta_graph('data/rf.ckpt.meta')
    saver.restore(sess, tf.train.latest_checkpoint('data/'))

    batch_x, batch_y = mnist.train.next_batch(10000)
    # 访问placeholders变量，并且创建feed-dict来作为placeholders的新值
    graph = tf.get_default_graph()
    X = graph.get_tensor_by_name("X:0")
    Y = graph.get_tensor_by_name("Y:0")
    feed_dict = {X: batch_x, Y: batch_y}
    predict = graph.get_tensor_by_name("pre:0")
    np.savetxt('data/expected.txt', batch_y, fmt='%01d')
    np.savetxt('data/predicted.txt', sess.run(predict, feed_dict), fmt='%01d')
    op_to_restore = graph.get_tensor_by_name("acc:0")

    print(sess.run(op_to_restore, feed_dict))
