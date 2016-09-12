#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
from input import get_input_vectors
from image_helpers import save
import numpy as np
from time import time


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
perm = perm + perm


grad3 = [[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
         [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
         [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]]


def fast_floor(x):
    if x > 0:
        return int(x)
    else:
        return int(x-1)


# Compute Perlin noise at coordinates x, y
def perlin2d(x, y):
    # Determine grid cell coordinates
    x0 = fast_floor(x)
    x1 = x0 + 1
    y0 = fast_floor(y)
    y1 = y0 + 1

    # Compute the vectors from the four points to the input point
    tx0 = x - x0
    tx1 = tx0 - 1
    ty0 = y - y0
    ty1 = ty0 - 1

    # Compute the gradient indices
    gi00 = perm[x0 + perm[y0]] % 12
    gi01 = perm[x1 + perm[y0]] % 12
    gi10 = perm[x0 + perm[y1]] % 12
    gi11 = perm[x1 + perm[y1]] % 12

    # Compute the dot-product between the vectors and the gradients
    v00 = grad3[gi00][0] * tx0 + grad3[gi00][1] * ty0
    v01 = grad3[gi01][0] * tx1 + grad3[gi01][1] * ty0
    v10 = grad3[gi10][0] * tx0 + grad3[gi10][1] * ty1
    v11 = grad3[gi11][0] * tx1 + grad3[gi11][1] * ty1

    # interpolate
    # wx = (3 - 2 * tx0) * tx0 * tx0  # this is the formula from the original version which produces artifacts
    wx = (10 - (15 - 6 * tx0) * tx0) * tx0 ** 3  # this is the improved formula from 2002
    v0 = v00 - wx * (v00 - v01)
    v1 = v10 - wx * (v10 - v11)
    # wy = (3 - 2 * ty0) * ty0 * ty0  # this is the formula from the original version which produces artifacts
    wy = (10 - (15 - 6 * ty0) * ty0) * ty0 ** 3  # this is the improved formula from 2002
    return v0 - wy * (v0 - v1)


def perlin3d(x, y, z):
    x0 = fast_floor(x)
    x1 = x0 + 1
    y0 = fast_floor(y)
    y1 = y0 + 1
    z0 = fast_floor(z)
    z1 = z0 + 1

    tx0 = x - x0
    tx1 = tx0 - 1
    ty0 = y - y0
    ty1 = ty0 - 1
    tz0 = z - z0
    tz1 = tz0 - 1

    gi000 = perm[x0 + perm[y0 + perm[z0]]] % 12
    gi001 = perm[x1 + perm[y0 + perm[z0]]] % 12
    gi010 = perm[x0 + perm[y1 + perm[z0]]] % 12
    gi100 = perm[x0 + perm[y0 + perm[z1]]] % 12
    gi011 = perm[x1 + perm[y1 + perm[z0]]] % 12
    gi101 = perm[x1 + perm[y0 + perm[z1]]] % 12
    gi110 = perm[x0 + perm[y1 + perm[z1]]] % 12
    gi111 = perm[x1 + perm[y1 + perm[z1]]] % 12

    v000 = grad3[gi000][0] * tx0 + grad3[gi000][1] * ty0 + grad3[gi000][2] * tz0
    v001 = grad3[gi001][0] * tx1 + grad3[gi001][1] * ty0 + grad3[gi001][2] * tz0
    v010 = grad3[gi010][0] * tx0 + grad3[gi010][1] * ty1 + grad3[gi010][2] * tz0
    v100 = grad3[gi100][0] * tx0 + grad3[gi100][1] * ty0 + grad3[gi100][2] * tz1
    v011 = grad3[gi011][0] * tx1 + grad3[gi011][1] * ty1 + grad3[gi011][2] * tz0
    v101 = grad3[gi101][0] * tx1 + grad3[gi101][1] * ty0 + grad3[gi101][2] * tz1
    v110 = grad3[gi110][0] * tx0 + grad3[gi110][1] * ty1 + grad3[gi110][2] * tz1
    v111 = grad3[gi111][0] * tx1 + grad3[gi111][1] * ty1 + grad3[gi111][2] * tz1

    wx = (10 - (15 - 6 * tx0) * tx0) * tx0 ** 3
    v00 = v000 - wx * (v000 - v001)
    v01 = v010 - wx * (v010 - v011)
    v10 = v100 - wx * (v100 - v101)
    v11 = v110 - wx * (v110 - v111)
    wy = (10 - (15 - 6 * ty0) * ty0) * ty0 ** 3
    v0 = v00 - wy * (v00 - v01)
    v1 = v10 - wy * (v10 - v11)
    wz = (10 - (15 - 6 * tz0) * tz0) * tz0 ** 3
    return v0 - wz * (v0 - v1)


if __name__ == "__main__":
    shape = (1080, 1920)
    phases = 1
    scaling = 100.0
    for frame in range(250):
        input_vectors = get_input_vectors(shape, phases, scaling, offset=(0, 0, frame/70))
        raw_noise = np.empty(input_vectors.shape[0], dtype=np.float32)
        start_time = time()
        for i in range(0, input_vectors.shape[0]):
            raw_noise[i] = perlin3d(input_vectors[i][0], input_vectors[i][1], input_vectors[i][2])
        print("The calculation took " + str(time() - start_time) + " seconds.")
        save(raw_noise, phases, shape, "classicPerlin3D" + "0"*(3-len(str(frame))), frame)
