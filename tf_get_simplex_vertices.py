#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import tensorflow as tf
import numpy as np

vertex_options = np.array([
    [1, 0, 0, 1, 1, 0],
    [1, 0, 0, 1, 0, 1],
    [0, 0, 1, 1, 0, 1],
    [0, 0, 1, 0, 1, 1],
    [0, 1, 0, 0, 1, 1],
    [0, 1, 0, 1, 1, 0]
], dtype=np.float32)

# Dimesions are: x0 >= y0, y0 >= z0, x0 >= z0
np_vertex_table = np.array([
    [[vertex_options[3], vertex_options[3]],
     [vertex_options[4], vertex_options[5]]],
    [[vertex_options[2], vertex_options[1]],
     [vertex_options[2], vertex_options[0]]]
], dtype=np.float32)


def get_simplex_vertices(offsets, vertex_table, length):
    vertex_table_x_index = tf.to_int32(offsets[:, 0] >= offsets[:, 1])
    vertex_table_y_index = tf.to_int32(offsets[:, 1] >= offsets[:, 2])
    vertex_table_z_index = tf.to_int32(offsets[:, 0] >= offsets[:, 2])
    index_list = tf.concat(1, [
        tf.reshape(tf.tile(tf.concat(1, [
            tf.expand_dims(vertex_table_x_index, 1),
            tf.expand_dims(vertex_table_y_index, 1),
            tf.expand_dims(vertex_table_z_index, 1),
        ]), [1, 6]), [6 * length, 3]),
        tf.expand_dims(tf.tile(tf.range(0, limit=6), [length]), 1)])
    vertices = tf.reshape(tf.gather_nd(vertex_table, index_list), [-1, 2, 3])
    return vertices


if __name__ == "__main__":
    vertex_table = tf.Variable(np_vertex_table, name='vertex_table')
    test_offsets = tf.Variable([[0.2, 0.1, 0.3], [0.03, 0.15, 0.12],
                                [0.34, 0.21, 0.31], [0.49, 0.0012, 0.237]], name="offsets")
    verts = get_simplex_vertices(test_offsets, vertex_table, 4)
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)
    print(sess.run(tf.to_int32(verts[:, 0, 1])))
