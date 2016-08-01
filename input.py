#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import numpy as np


def get_input_vectors(shape=(512, 512), phases=5, scaling=200, offset=(0.0, 0.0, 1.7)):
    x = np.repeat(offset[0] + np.arange(shape[0]) / scaling, shape[1] * phases).reshape(
        (shape[0], shape[1], phases)) * np.power(2, np.arange(phases))
    y = np.repeat(np.tile(offset[1] + np.arange(shape[1]) / scaling, shape[0]).reshape(
        (shape[0], shape[1], 1)), phases, axis=2) * np.power(2, np.arange(phases))
    z = np.tile(offset[2] + 10 * np.arange(phases), shape[0] * shape[1]).reshape((shape[0], shape[1], phases, 1))
    x = x.reshape((shape[0], shape[1], phases, 1))
    y = y.reshape((shape[0], shape[1], phases, 1))
    return np.append(np.append(x, y, axis=3), z, axis=3).reshape((shape[0] * shape[1] * phases, 3)).astype(np.float32)


if __name__ == "__main__":
    print(get_input_vectors())
