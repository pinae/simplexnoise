#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import numpy as np
from PIL import Image
from time import time


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

# Dimesions are: x0 >= y0, y0 >= z0, x0 >= z0
vertices_table = np.array([
    [[[vertices_options[3]], [vertices_options[3]]],
     [[vertices_options[4]], [vertices_options[5]]]],
    [[[vertices_options[2]], [vertices_options[1]]],
     [[vertices_options[2]], [vertices_options[0]]]]
], dtype=np.uint8)


def select_simplex(ox, oy, oz):
    return vertices_table[
        int(ox >= oy),
        int(oy >= oz),
        int(ox >= oz)].reshape((2, 3))


def calculate_gradient_contribution(offsets, gis, gradient_map):
    t = 0.5 - offsets[:, 0] ** 2 - offsets[:, 1] ** 2 - offsets[:, 2] ** 2
    return np.where(
        t < 0,
        0.0,
        t ** 4 * np.array([np.dot(gradient_map[gis[i].astype('int32')], offset) for i, offset in enumerate(offsets)]))


def matrix_noise3d(input_vectors, perm, grad3):
    skew_factors = (input_vectors[:, 0] + input_vectors[:, 1] + input_vectors[:, 2]) * 1.0 / 3.0
    skewed_vectors = np.floor(input_vectors + skew_factors[:, np.newaxis])
    unskew_factors = (skewed_vectors[:, 0] + skewed_vectors[:, 1] + skewed_vectors[:, 2]) * 1.0 / 6.0
    offsets_0 = input_vectors - (skewed_vectors - unskew_factors[:, np.newaxis])
    simplex_vertices = np.array([select_simplex(offset[0], offset[1], offset[2]) for offset in offsets_0])
    offsets_1 = offsets_0 - simplex_vertices[:, 0] + 1.0 / 6.0
    offsets_2 = offsets_0 - simplex_vertices[:, 1] + 1.0 / 3.0
    offsets_3 = offsets_0 - 0.5
    masked_skewed_vectors = np.bitwise_and(skewed_vectors.astype(np.int), 255)
    gi0s = perm[masked_skewed_vectors[:, 0] + perm[
        masked_skewed_vectors[:, 1] + perm[
            masked_skewed_vectors[:, 2]].astype(np.int)].astype(np.int)] % 12
    gi1s = perm[masked_skewed_vectors[:, 0] + simplex_vertices[:, 0, 0] + perm[
        masked_skewed_vectors[:, 1] + simplex_vertices[:, 0, 1] + perm[
            masked_skewed_vectors[:, 2] + simplex_vertices[:, 0, 2]].astype(np.int)].astype(np.int)] % 12
    gi2s = perm[masked_skewed_vectors[:, 0] + simplex_vertices[:, 1, 0] + perm[
        masked_skewed_vectors[:, 1] + simplex_vertices[:, 1, 1] + perm[
            masked_skewed_vectors[:, 2] + simplex_vertices[:, 1, 2]].astype(np.int)].astype(np.int)] % 12
    gi3s = perm[masked_skewed_vectors[:, 0] + 1 + perm[
        masked_skewed_vectors[:, 1] + 1 + perm[
            masked_skewed_vectors[:, 2] + 1].astype(np.int)].astype(np.int)] % 12
    n0s = calculate_gradient_contribution(offsets_0, gi0s, grad3)
    n1s = calculate_gradient_contribution(offsets_1, gi1s, grad3)
    n2s = calculate_gradient_contribution(offsets_2, gi2s, grad3)
    n3s = calculate_gradient_contribution(offsets_3, gi3s, grad3)
    return 23.0 * (n0s + n1s + n2s + n3s)


if __name__ == "__main__":
    input_vectors = np.zeros((256 * 256, 3), dtype=np.float32)
    for y in range(0, 256):
        for x in range(0, 256):
            input_vectors[y * 256 + x] = [x / 80.0, y / 80.0, 1.7]
    start_time = time()
    raw_noise = matrix_noise3d(input_vectors, np_perm, np_grad3)
    print("The calculation took " + str(time() - start_time) + " seconds.")
    arr = np.empty((256, 256, 3), dtype=np.uint8)
    for y in range(0, 256):
        for x in range(0, 256):
            val = raw_noise[x + y * 256]
            val = int(np.floor((val + 1.0) * 128))
            arr[x, y, 0] = val
            arr[x, y, 1] = val
            arr[x, y, 2] = val
    image = Image.fromarray(arr)
    image.show()
