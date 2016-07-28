#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from PIL import Image
from time import time
from tf_get_simplex_vertices import get_simplex_vertices
from tf_map_gradient import map_gradients


np_perm = [151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99,
           37, 240, 21, 10, 23, 190, 6, 148, 247, 120, 234, 75, 0, 26, 197, 62, 94, 252, 219, 203, 117, 35, 11, 32,
           57, 177, 33, 88, 237, 149, 56, 87, 174, 20, 125, 136, 171, 168, 68, 175, 74, 165, 71, 134, 139, 48, 27,
           166, 77, 146, 158, 231, 83, 111, 229, 122, 60, 211, 133, 230, 220, 105, 92, 41, 55, 46, 245, 40, 244, 102,
           143, 54, 65, 25, 63, 161, 1, 216, 80, 73, 209, 76, 132, 187, 208, 89, 18, 169, 200, 196, 135, 130, 116,
           188, 159, 86, 164, 100, 109, 198, 173, 186, 3, 64, 52, 217, 226, 250, 124, 123, 5, 202, 38, 147, 118, 126,
           255, 82, 85, 212, 207, 206, 59, 227, 47, 16, 58, 17, 182, 189, 28, 42, 223, 183, 170, 213, 119, 248, 152,
           2, 44, 154, 163, 70, 221, 153, 101, 155, 167, 43, 172, 9, 129, 22, 39, 253, 19, 98, 108, 110, 79, 113, 224,
           232, 178, 185, 112, 104, 218, 246, 97, 228, 251, 34, 242, 193, 238, 210, 144, 12, 191, 179, 162, 241, 81,
           51, 145, 235, 249, 14, 239, 107, 49, 192, 214, 31, 181, 199, 106, 157, 184, 84, 204, 176, 115, 121, 50,
           45, 127, 4, 150, 254, 138, 236, 205, 93, 222, 114, 67, 29, 24, 72, 243, 141, 128, 195, 78, 66, 215, 61,
           156, 180]
np_perm = np.array(np_perm + np_perm, dtype=np.int32)

np_grad3 = np.array([[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
                     [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
                     [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]], dtype=np.float32)

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


def calculate_gradient_contribution(offsets, gis, gradient_map, length):
    t = 0.5 - offsets[:, 0] ** 2 - offsets[:, 1] ** 2 - offsets[:, 2] ** 2
    mapped_gis = map_gradients(gradient_map, gis, length)
    dot_products = tf.reduce_sum(mapped_gis * offsets, 1)
    return tf.to_float(tf.greater_equal(t, 0)) * t ** 4 * dot_products


def noise3d(input_vectors, perm, grad3, vertex_table, length):
    skew_factors = (input_vectors[:, 0] + input_vectors[:, 1] + input_vectors[:, 2]) * 1.0 / 3.0
    skewed_vectors = tf.floor(input_vectors + tf.expand_dims(skew_factors, 1))
    unskew_factors = (skewed_vectors[:, 0] + skewed_vectors[:, 1] + skewed_vectors[:, 2]) * 1.0 / 6.0
    offsets_0 = input_vectors - (skewed_vectors - tf.expand_dims(unskew_factors, 1))
    simplex_vertices = get_simplex_vertices(offsets_0, vertex_table, length)
    offsets_1 = offsets_0 - simplex_vertices[:, 0, :] + 1.0 / 6.0
    offsets_2 = offsets_0 - simplex_vertices[:, 1, :] + 1.0 / 3.0
    offsets_3 = offsets_0 - 0.5
    masked_skewed_vectors = tf.to_int32(skewed_vectors) % 256
    gi0s = tf.gather_nd(
        perm,
        tf.expand_dims(masked_skewed_vectors[:, 0], 1) +
        tf.expand_dims(tf.gather_nd(
            perm,
            tf.expand_dims(masked_skewed_vectors[:, 1], 1) +
            tf.expand_dims(tf.gather_nd(
                perm,
                tf.expand_dims(masked_skewed_vectors[:, 2], 1)), 1)), 1)
    ) % 12
    gi1s = tf.gather_nd(
        perm,
        tf.expand_dims(masked_skewed_vectors[:, 0], 1) +
        tf.expand_dims(tf.to_int32(simplex_vertices[:, 0, 0]), 1) +
        tf.expand_dims(tf.gather_nd(
            perm,
            tf.expand_dims(masked_skewed_vectors[:, 1], 1) +
            tf.expand_dims(tf.to_int32(simplex_vertices[:, 0, 1]), 1) +
            tf.expand_dims(tf.gather_nd(
                perm,
                tf.expand_dims(masked_skewed_vectors[:, 2], 1) +
                tf.expand_dims(tf.to_int32(simplex_vertices[:, 0, 2]), 1)), 1)), 1)
    ) % 12
    gi2s = tf.gather_nd(
        perm,
        tf.expand_dims(masked_skewed_vectors[:, 0], 1) +
        tf.expand_dims(tf.to_int32(simplex_vertices[:, 1, 0]), 1) +
        tf.expand_dims(tf.gather_nd(
            perm,
            tf.expand_dims(masked_skewed_vectors[:, 1], 1) +
            tf.expand_dims(tf.to_int32(simplex_vertices[:, 1, 1]), 1) +
            tf.expand_dims(tf.gather_nd(
                perm,
                tf.expand_dims(masked_skewed_vectors[:, 2], 1) +
                tf.expand_dims(tf.to_int32(simplex_vertices[:, 1, 2]), 1)), 1)), 1)
    ) % 12
    gi3s = tf.gather_nd(
        perm,
        tf.expand_dims(masked_skewed_vectors[:, 0], 1) +
        1 +
        tf.expand_dims(tf.gather_nd(
            perm,
            tf.expand_dims(masked_skewed_vectors[:, 1], 1) +
            1 +
            tf.expand_dims(tf.gather_nd(
                perm,
                tf.expand_dims(masked_skewed_vectors[:, 2], 1) +
                1), 1)), 1)
    ) % 12
    n0s = calculate_gradient_contribution(offsets_0, gi0s, grad3, length)
    n1s = calculate_gradient_contribution(offsets_1, gi1s, grad3, length)
    n2s = calculate_gradient_contribution(offsets_2, gi2s, grad3, length)
    n3s = calculate_gradient_contribution(offsets_3, gi3s, grad3, length)
    return 23.0 * tf.squeeze(
        tf.expand_dims(n0s, 1) + tf.expand_dims(n1s, 1) + tf.expand_dims(n2s, 1) + tf.expand_dims(n3s, 1))


if __name__ == "__main__":
    arr = np.empty((512, 512, 3), dtype=np.uint8)
    input_vectors = np.zeros((arr.shape[1] * arr.shape[0], 3), dtype=np.float32)
    for y in range(0, arr.shape[0]):
        for x in range(0, arr.shape[1]):
            input_vectors[y * arr.shape[1] + x] = [x / 80.0, y / 80.0, 1.7]
    perm = tf.Variable(np_perm, name='perm')
    grad3 = tf.Variable(np_grad3, name='grad3')
    vl = tf.Variable(input_vectors, name='vector_list')
    vertex_table = tf.Variable(np_vertex_table, name='vertex_table')
    raw_noise = noise3d(vl, perm, grad3, vertex_table, input_vectors.shape[0])
    init = tf.initialize_all_variables()
    sess = tf.Session()
    start_time = time()
    sess.run(init)
    raw_noise = sess.run(noise3d(input_vectors, np_perm, np_grad3, np_vertex_table, input_vectors.shape[0]))
    print("The calculation took " + str(time() - start_time) + " seconds.")
    for y in range(0, arr.shape[0]):
        for x in range(0, arr.shape[1]):
            val = raw_noise[x + y * arr.shape[1]]
            val = int(np.floor((val + 1.0) * 128))
            arr[y, x, 0] = val
            arr[y, x, 1] = val
            arr[y, x, 2] = val
    image = Image.fromarray(arr)
    image.show()
