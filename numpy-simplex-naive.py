#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import numpy as np
from time import time
from image_helpers import show

perm = [151, 160, 137, 91, 90, 15, 131, 13, 201, 95, 96, 53, 194, 233, 7, 225, 140, 36, 103, 30, 69, 142, 8, 99,
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
perm = np.array(perm + perm)

grad3 = np.array([[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
                  [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
                  [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]])

vertices_options = [
    [[1, 0, 0], [1, 1, 0]],
    [[1, 0, 0], [1, 0, 1]],
    [[0, 0, 1], [1, 0, 1]],
    [[0, 0, 1], [0, 1, 1]],
    [[0, 1, 0], [0, 1, 1]],
    [[0, 1, 0], [1, 1, 0]]
]

# Dimesions are: x0 >= y0, y0 >= z0, x0 >= z0
vertices_table = np.array([
    [[[vertices_options[3]], [vertices_options[3]]],
     [[vertices_options[4]], [vertices_options[5]]]],
    [[[vertices_options[2]], [vertices_options[1]]],
     [[vertices_options[2]], [vertices_options[0]]]]
])


def np_noise3d(v):
    skew_factor = np.sum(v) * 1.0/3.0
    skewed_v = np.floor(v + skew_factor)
    unskew_factor = np.sum(skewed_v) * 1.0/6.0
    offset_0 = v - (skewed_v - unskew_factor)
    vertices = vertices_table[
        int(offset_0[0] >= offset_0[1]),
        int(offset_0[1] >= offset_0[2]),
        int(offset_0[0] >= offset_0[2])].reshape((2, 3))
    offsets = [offset_0,
               offset_0 - vertices[0] + 1.0/6.0,
               offset_0 - vertices[1] + 1.0/3.0,
               offset_0 - 0.5]
    skewed_v_and = np.bitwise_and(skewed_v.astype(np.int), 255)
    gi = np.array([perm[skewed_v_and[0] + perm[
                       skewed_v_and[1] + perm[
                          skewed_v_and[2]]]] % 12,
                   perm[skewed_v_and[0] + vertices[0][0] + perm[
                       skewed_v_and[1] + vertices[0][1] + perm[
                           skewed_v_and[2] + vertices[0][2]]]] % 12,
                   perm[skewed_v_and[0] + vertices[1][0] + perm[
                       skewed_v_and[1] + vertices[1][1] + perm[
                           skewed_v_and[2] + vertices[1][2]]]] % 12,
                   perm[skewed_v_and[0] + 1 + perm[
                       skewed_v_and[1] + 1 + perm[
                           skewed_v_and[2] + 1]]] % 12])
    n = np.zeros(4, dtype=np.float)
    for i in range(4):
        t = 0.5 + np.sum(-1 * offsets[i]**2)
        if t >= 0:
            t *= t
            n[i] = t**2 * np.dot(grad3[gi[i]], offsets[i])
    return 23.0 * np.sum(n)


if __name__ == "__main__":
    arr = np.empty((512, 512, 3), dtype=np.uint8)
    phases = 5
    scaling = 200.0
    input_vectors = np.zeros((arr.shape[1] * arr.shape[0] * phases, 3), dtype=np.float32)
    for y in range(0, arr.shape[0]):
        for x in range(0, arr.shape[1]):
            for phase in range(phases):
                input_vectors[y * arr.shape[1] * phases + x * phases + phase] = \
                    [x / scaling * np.power(2, phase), y / scaling * np.power(2, phase), 1.7 + 10 * phase]
    raw_noise = np.empty(input_vectors.shape[0], dtype=np.float32)
    start_time = time()
    for i in range(0, input_vectors.shape[0]):
        raw_noise[i] = np_noise3d(input_vectors[i])
    print("The calculation took " + str(time() - start_time) + " seconds.")
    show(raw_noise, phases, arr.shape)
