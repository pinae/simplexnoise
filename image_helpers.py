#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
import numpy as np
from PIL import Image


def sum_phases(raw_noise, phases, shape):
    val = np.floor((np.sum(
        raw_noise.reshape((shape[0], shape[1], phases)) / np.power(
            2, np.arange(phases)), axis=2) + 1) * 128).astype(np.uint8).reshape((shape[0], shape[1], 1))
    return np.append(val, np.append(val, val, axis=2), axis=2)


def show(image_data):
    image = Image.fromarray(image_data)
    image.show()


def save(image_data, filename="image", index=0):
    image = Image.fromarray(image_data)
    image.save(filename + str(index) + ".jpg")
