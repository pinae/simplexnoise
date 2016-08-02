#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import numpy as np
import theano.tensor as T
import theano
from time import time
from input import get_input_vectors
from image_helpers import show


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

vertices_options = np.array([
    [[1, 0, 0], [1, 1, 0]],
    [[1, 0, 0], [1, 0, 1]],
    [[0, 0, 1], [1, 0, 1]],
    [[0, 0, 1], [0, 1, 1]],
    [[0, 1, 0], [0, 1, 1]],
    [[0, 1, 0], [1, 1, 0]]
], dtype=np.int32)

# Dimensions are: x0 >= y0, y0 >= z0, x0 >= z0
vertices_table = np.array([
    [[[vertices_options[3]], [vertices_options[3]]],
     [[vertices_options[4]], [vertices_options[5]]]],
    [[[vertices_options[2]], [vertices_options[1]]],
     [[vertices_options[2]], [vertices_options[0]]]]
], dtype=np.uint8)


def t_noise3d(v, perm, grad3):
    x = v[0]
    y = v[1]
    z = v[2]
    skew_factor = (x + y + z) * 1.0 / 3.0
    i = T.floor(x + skew_factor)
    j = T.floor(y + skew_factor)
    k = T.floor(z + skew_factor)
    unskew_factor = (i + j + k) * 1.0 / 6.0
    x0 = x - (i - unskew_factor)
    y0 = y - (j - unskew_factor)
    z0 = z - (k - unskew_factor)
    vertices = T.switch(T.ge(x0, y0),
                        T.switch(T.ge(y0, z0), vertices_options[0],
                                 T.switch(T.ge(x0, z0), vertices_options[1],
                                          vertices_options[2])),
                        T.switch(T.lt(y0, z0), vertices_options[3],
                                 T.switch(T.lt(x0, z0), vertices_options[4],
                                          vertices_options[5]))
                        )
    x1 = x0 - vertices[0][0] + 1.0 / 6.0
    y1 = y0 - vertices[0][1] + 1.0 / 6.0
    z1 = z0 - vertices[0][2] + 1.0 / 6.0
    x2 = x0 - vertices[1][0] + 1.0 / 3.0
    y2 = y0 - vertices[1][1] + 1.0 / 3.0
    z2 = z0 - vertices[1][2] + 1.0 / 3.0
    x3 = x0 - 0.5
    y3 = y0 - 0.5
    z3 = z0 - 0.5
    ii = T.bitwise_and(i.astype('int32'), 255)
    jj = T.bitwise_and(j.astype('int32'), 255)
    kk = T.bitwise_and(k.astype('int32'), 255)
    gi0 = perm[ii + perm[
            jj + perm[
                kk].astype('int32')].astype('int32')] % 12
    gi1 = perm[ii + vertices[0][0] + perm[
            jj + vertices[0][1] + perm[
                kk + vertices[0][2]].astype('int32')].astype('int32')] % 12
    gi2 = perm[ii + vertices[1][0] + perm[
            jj + vertices[1][1] + perm[
                kk + vertices[1][2]].astype('int32')].astype('int32')] % 12
    gi3 = perm[ii + 1 + perm[
            jj + 1 + perm[
                kk + 1].astype('int32')].astype('int32')] % 12
    t0 = 0.5 - x0 ** 2 - y0 ** 2 - z0 ** 2
    n0 = T.switch(
        T.lt(t0, 0),
        0.0,
        t0 ** 4 * T.dot(grad3[gi0.astype('int32')], [x0, y0, z0]))
    t1 = 0.5 - x1 ** 2 - y1 ** 2 - z1 ** 2
    n1 = T.switch(
        T.lt(t1, 0),
        0.0,
        t1 ** 4 * T.dot(grad3[gi1.astype('int32')], [x1, y1, z1])),
    t2 = 0.5 - x2 ** 2 - y2 ** 2 - z2 ** 2
    n2 = T.switch(
        T.lt(t2, 0),
        0.0,
        t2 ** 4 * T.dot(grad3[gi2.astype('int32')], [x2, y2, z2]))
    t3 = 0.5 - x3 ** 2 - y3 ** 2 - z3 ** 2
    n3 = T.switch(
        T.lt(t3, 0),
        0.0,
        t3 ** 4 * T.dot(grad3[gi3.astype('int32')], [x3, y3, z3]))
    return 23.0 * (n0 + n1 + n2 + n3)


if __name__ == "__main__":
    theano.config.openmp = True
    theano.config.openmp_elemwise_minsize = 200
    perm = T.vector('perm', dtype='int32')
    grad3 = T.matrix('grad3', dtype='float32')
    vl = T.matrix('vl', dtype='float32')
    output, updates = theano.map(fn=t_noise3d,
                                 sequences=[vl],
                                 non_sequences=[perm, grad3],
                                 name="noise_all_pixels")
    simplex_noise = theano.function([vl, perm, grad3], output)
    print("Compiled")
    shape = (512, 512)
    phases = 5
    scaling = 200.0
    input_vectors = get_input_vectors(shape, phases, scaling)
    start_time = time()
    raw_noise = simplex_noise(input_vectors, np_perm, np_grad3)
    print("The calculation took " + str(time() - start_time) + " seconds.")
    show(raw_noise, phases, shape)
