#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
from PIL import Image
import numpy as np
from math import floor


def get_perm():
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
    for i in range(256, 512):
        perm.append(perm[i & 255])
    return perm


def grad3(index):
    return [[1, 1, 0],
            [-1, 1, 0],
            [1, -1, 0],
            [-1, -1, 0],
            [1, 0, 1],
            [-1, 0, 1],
            [1, 0, -1],
            [-1, 0, -1],
            [0, 1, 1],
            [0, -1, 1],
            [0, 1, -1],
            [0, -1, -1]][index]


def skew(x, y, z):
    skewfactor = (x + y + z) / 3.0
    i = int(floor(x + skewfactor))
    j = int(floor(y + skewfactor))
    k = int(floor(z + skewfactor))
    return i, j, k


def calculate_origin_offset(x, y, z, skewed_simplex_origin):
    unskewfactor = 1 / 6.0
    t = (x + y + z) * unskewfactor
    x0 = x - (skewed_simplex_origin[0] - t)
    y0 = y - (skewed_simplex_origin[1] - t)
    z0 = z - (skewed_simplex_origin[2] - t)
    return x0, y0, z0


def calculate_vertex_offsets(x0, y0, z0, i1, j1, k1, i2, j2, k2):
    x1 = x0 - i1 + 1 / 6.0
    y1 = y0 - j1 + 1 / 6.0
    z1 = z0 - k1 + 1 / 6.0
    x2 = x0 - i2 + 1 / 3.0
    y2 = y0 - j2 + 1 / 3.0
    z2 = z0 - k2 + 1 / 3.0
    x3 = x0 - 0.5
    y3 = y0 - 0.5
    z3 = z0 - 0.5
    return x1, y1, z1, x2, y2, z2, x3, y3, z3


def find_simplex(x0, y0, z0):
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
    return i1, j1, k1, i2, j2, k2


def get_vertex_gradient_indices(perm, skewed_simplex_origin, i1, j1, k1, i2, j2, k2):
    ii = skewed_simplex_origin[0] & 255
    jj = skewed_simplex_origin[1] & 255
    kk = skewed_simplex_origin[2] & 255
    gi0 = perm[ii + perm[jj + perm[kk]]] % 12
    gi1 = perm[ii + i1 + perm[jj + j1 + perm[kk + k1]]] % 12
    gi2 = perm[ii + i2 + perm[jj + j2 + perm[kk + k2]]] % 12
    gi3 = perm[ii + 1 + perm[jj + 1 + perm[kk + 1]]] % 12
    return gi0, gi1, gi2, gi3


def calculate_gradient_contribution(xd, yd, zd, gi):
    n = [0.0, 0.0, 0.0, 0.0]
    for vertex_no in range(4):
        t = 0.5 - xd[vertex_no] * xd[vertex_no] - yd[vertex_no] * yd[vertex_no] - zd[vertex_no] * zd[vertex_no]
        if t >= 0:
            t *= t
            n[vertex_no] = t * t * np.dot(grad3(gi[vertex_no]), [xd[vertex_no], yd[vertex_no], zd[vertex_no]])
    return 32.0 * np.sum(n)


def noise(x, y, z):
    perm = get_perm()
    skewed_simplex_origin = skew(x, y, z)
    x0, y0, z0 = calculate_origin_offset(x, y, z, skewed_simplex_origin)
    i1, j1, k1, i2, j2, k2 = find_simplex(x0, y0, z0)
    x1, y1, z1, x2, y2, z2, x3, y3, z3 = calculate_vertex_offsets(x0, y0, z0, i1, j1, k1, i2, j2, k2)
    gi = get_vertex_gradient_indices(perm, skewed_simplex_origin, i1, j1, k1, i2, j2, k2)
    return calculate_gradient_contribution((x0, x1, x2, x3), (y0, y1, y2, y3), (z0, z1, z2, z3), gi)


if __name__ == "__main__":
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    for y in range(0, 256):
        for x in range(0, 256):
            val = noise(x/100.0, y/120.0, 0.0)
            val = max(0, min(255, int(round((val + 0.5) * 128))))
            arr[x, y, 0] = val
            arr[x, y, 1] = val
            arr[x, y, 2] = val
    image = Image.fromarray(arr)
    image.show()
