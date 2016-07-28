#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import tensorflow as tf
import numpy as np


np_grad3 = np.array([[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
                     [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
                     [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]], dtype=np.float32)


def map_gradients(gradient_map, gis, length):
    index_tensor = tf.reshape(tf.concat(1, [
        tf.reshape(tf.tile(tf.expand_dims(gis, 1), [1, 3]), [length * 3, 1]),
        tf.expand_dims(tf.tile(tf.range(0, limit=3), [length]), 1)
    ]), [length, 3, 2])
    return tf.gather_nd(gradient_map, index_tensor)


if __name__ == "__main__":
    gradient_map = tf.Variable(np_grad3, name='vertex_table')
    gis = tf.Variable([0, 3, 7, 2, 9, 11, 7, 4], name="gis")
    gradients = map_gradients(gradient_map, gis, 8)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    print(sess.run(gradients))
