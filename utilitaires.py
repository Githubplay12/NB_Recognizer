import numpy as np
import cv2
import argparse
from matplotlib import pyplot as plt

# Goal is to move an image either left, right, up or down or a cimbinaison of the above
def translate(image, x, y):
    # x = negative if left, positive if right / y = positive if down, negative if up
    # M is the translation matrix, np expects a float
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    return shifted

# Rotation method to avoid too much writing
def rotate(image, degrees, center = None, scale = 1.0):
    # Positive degrees = left rotation, Negative degrees = right rotation
    height, width = image.shape[:2]

    if not center:
        center = (width // 2, height // 2)

    M = cv2.getRotationMatrix2D(center, degrees, 1.0)
    rotated = cv2.warpAffine(image, M, (width, height))
    return rotated

# Resizing method, INTER_AREA only for now
def resize(image, desiredsize, bywhat = 'width', inter = cv2.INTER_AREA):
    h, w = image.shape[:2]
    resized = ''
    dim = ''
    if bywhat != 'height':
        r = float(desiredsize) / float(w)
        dim = (desiredsize, int(h * r))
    else:
        r = float(desiredsize) / float(h)
        dim = (int(w * r), desiredsize)

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized

# Flipping is too simple to write a method for

# Cropping is too simple

# Histogram plotting
def plot_histogram(image, title, mask = None):
    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    plt.figure()
    plt.title(title)
    plt.xlabel('# of Bins')
    plt.ylabel('# of Pixels')

    for chan, color in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])



