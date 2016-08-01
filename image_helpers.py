#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import numpy as np
from PIL import Image


def show(raw_noise, phases, shape):
    val = np.floor((np.sum(
        raw_noise.reshape((shape[0], shape[1], phases)) / np.power(
            2, np.arange(phases)), axis=2) + 1) * 128).astype(np.uint8).reshape((shape[0], shape[1], 1))
    arr = np.append(val, np.append(val, val, axis=2), axis=2)
    image = Image.fromarray(arr)
    image.show()
