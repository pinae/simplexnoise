#!/usr/bin/python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
from PIL import Image
import numpy as np
from random import randint


def noise(x, y, z):
    return randint(0, 256)

if __name__ == "__main__":
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    for y in range(0, 256):
        for x in range(0, 256):
            val = noise(x, y, 0)
            arr[x, y, 0] = val
            arr[x, y, 1] = val
            arr[x, y, 2] = val
    image = Image.fromarray(arr)
    image.show()
