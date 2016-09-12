#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import tensorflow as tf


def tf_repeat(x, num_repeats):
    x = tf.reshape(x, [-1, 1])
    x = tf.tile(x, [1, num_repeats])
    return tf.reshape(x, [-1])


def get_input_vectors(shape, phases, scaling, offset):
    x = tf.reshape(tf_repeat(offset[0] + tf.linspace(0.0, tf.to_float(shape[0] - 1), shape[0]) / scaling,
                             shape[1] * phases),
                   [shape[0], shape[1], phases]) * tf.pow(2.0, tf.linspace(0.0, tf.to_float(phases - 1), phases))
    y = tf.reshape(tf_repeat(tf.tile(
        offset[1] + tf.linspace(0.0, tf.to_float(shape[1] - 1), shape[1]) / scaling,
        [shape[0]]
    ), phases), [shape[0], shape[1], phases]) * tf.pow(2.0, tf.linspace(0.0, tf.to_float(phases - 1), phases))
    z = tf.reshape(
        tf.tile(offset[2] + 10 * tf.linspace(0.0, tf.to_float(phases - 1), phases), [shape[0] * shape[1]]),
        [shape[0], shape[1], phases, 1])
    x = tf.reshape(x, [shape[0], shape[1], phases, 1])
    y = tf.reshape(y, [shape[0], shape[1], phases, 1])
    return tf.reshape(tf.concat(3, [x, y, z]), [shape[0] * shape[1] * phases, 3])
