#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import numpy as np
from math import sqrt
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
perm = perm + perm

grad3 = [[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
         [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
         [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1]]


def fast_floor(x):
    if x > 0:
        return int(x)
    else:
        return int(x-1)


def noise3d(x, y, z):
    global perm
    global grad3

    F3 = 1.0/3.0
    s = (x+y+z)*F3
    i = fast_floor(x+s)
    j = fast_floor(y+s)
    k = fast_floor(z+s)

    G3 = 1.0/6.0
    t = (i+j+k)*G3
    x0 = x-(i-t)
    y0 = y-(j-t)
    z0 = z-(k-t)

    if x0 >= y0:
        if y0 >= z0:
            i1, j1, k1, i2, j2, k2 = (1, 0, 0, 1, 1, 0)
        elif x0 >= z0:
            i1, j1, k1, i2, j2, k2 = (1, 0, 0, 1, 0, 1)
        else:
            i1, j1, k1, i2, j2, k2 = (0, 0, 1, 1, 0, 1)
    else:
        if y0 < z0:
            i1, j1, k1, i2, j2, k2 = (0, 0, 1, 0, 1, 1)
        elif x0 < z0:
            i1, j1, k1, i2, j2, k2 = (0, 1, 0, 0, 1, 1)
        else:
            i1, j1, k1, i2, j2, k2 = (0, 1, 0, 1, 1, 0)

    x1 = x0-i1+G3
    y1 = y0-j1+G3
    z1 = z0-k1+G3
    x2 = x0-i2+2.0*G3
    y2 = y0-j2+2.0*G3
    z2 = z0-k2+2.0*G3
    x3 = x0-1.0+3.0*G3
    y3 = y0-1.0+3.0*G3
    z3 = z0-1.0+3.0*G3

    ii = i & 255
    jj = j & 255
    kk = k & 255
    gi0 = perm[ii + perm[jj + perm[kk]]] % 12
    gi1 = perm[ii + i1 + perm[jj + j1 + perm[kk + k1]]] % 12
    gi2 = perm[ii + i2 + perm[jj + j2 + perm[kk + k2]]] % 12
    gi3 = perm[ii + 1 + perm[jj + 1 + perm[kk + 1]]] % 12

    t0 = 0.5 - x0 * x0 - y0 * y0 - z0 * z0
    if t0 < 0:
        n0 = 0.0
    else:
        t0 *= t0
        n0 = t0 * t0 * np.dot(grad3[gi0], [x0, y0, z0])
    t1 = 0.5 - x1 * x1 - y1 * y1 - z1 * z1
    if t1 < 0:
        n1 = 0.0
    else:
        t1 *= t1
        n1 = t1 * t1 * np.dot(grad3[gi1], [x1, y1, z1])
    t2 = 0.5 - x2 * x2 - y2 * y2 - z2 * z2
    if t2 < 0:
        n2 = 0.0
    else:
        t2 *= t2
        n2 = t2 * t2 * np.dot(grad3[gi2], [x2, y2, z2])
    t3 = 0.5 - x3 * x3 - y3 * y3 - z3 * z3
    if t3 < 0:
        n3 = 0.0
    else:
        t3 *= t3
        n3 = t3 * t3 * np.dot(grad3[gi3], [x3, y3, z3])

    return 23.0 * (n0 + n1 + n2 + n3)


def noise2d(x, y):
    global perm
    global grad3
    F2 = 0.5 * (sqrt(3.0) - 1.0)
    s = (x + y) * F2
    i, j = (fast_floor(x + s), fast_floor(y + s))
    G2 = (3.0 - sqrt(3.0)) / 6.0
    t = (i + j) * G2
    dist = (x - (i - t), y - (j - t))
    if dist[0] > dist[1]:
        second_corner = (1, 0)
    else:
        second_corner = (0, 1)
    offsets = [(dist[0]-second_corner[0]+G2, dist[1]-second_corner[1]+G2),
               (dist[0]-1.0+2.0*G2, dist[1]-1.0+2.0*G2)]
    ii, jj = (i & 255, j & 255)
    gi = [perm[ii+perm[jj]] % 12,
          perm[ii+second_corner[0]+perm[jj+second_corner[1]]] % 12,
          perm[ii+1+perm[jj+1]] % 12]
    n = [0.0, 0.0, 0.0]
    t0 = 0.5-dist[0]**2-dist[1]**2
    if t0 >= 0:
        t0 *= t0
        n[0] = t0*t0*np.dot(grad3[gi[0]][:2], dist)
    t1 = 0.5 - offsets[0][0] ** 2 - offsets[0][1] ** 2
    if t1 >= 0:
        t1 *= t1
        n[1] = t1 * t1 * np.dot(grad3[gi[1]][:2], offsets[0])
    t2 = 0.5 - offsets[1][0] ** 2 - offsets[1][1] ** 2
    if t2 >= 0:
        t2 *= t2
        n[2] = t2 * t2 * np.dot(grad3[gi[2]][:2], offsets[1])
    return 70.0*(n[0]+n[1]+n[2])


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
        raw_noise[i] = noise3d(input_vectors[i][0], input_vectors[i][1], input_vectors[i][2])
    print("The calculation took " + str(time() - start_time) + " seconds.")
    show(raw_noise, phases, arr.shape)
