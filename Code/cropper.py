#!/usr/bin/env python3

import math
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import glob
import os.path

def sort_contours(cnts_):

    boundingBoxes = [cv.boundingRect(c) for c in cnts_]
    (cnts_, boundingBoxes) = zip(*sorted(zip(cnts_, boundingBoxes),
                                        key=lambda b: b[1][3]))
        # return the list of sorted contours and bounding boxes
    return (cnts_, boundingBoxes)

def cropper(img):
    (w, h) = img.shape
    w = int(w/2)
    h = int(h/2)
    limg = img[w:, :h]
    th = cv.adaptiveThreshold(limg, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv.THRESH_BINARY_INV, 11, 11)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    th = cv.morphologyEx(th, cv.MORPH_OPEN, kernel)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 3))
    dilated = cv.dilate(th, kernel, iterations=1)
    cnts, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    sortedcnts, BB = sort_contours(cnts)

    print(BB[-1], BB[-1][3]/BB[-1][2])

    cropped = limg[BB[-1][1]-3: BB[-1][1]+BB[-1][3]+3, BB[-1][0]+3:BB[-1][0]+BB[-1][2]-3]

    # cv.rectangle(limg, (BB[-1][0]+3, BB[-1][1]-3), (BB[-1][0]+BB[-1][2]-3, BB[-1][1]+BB[-1][3]+3), 0, 2 );
    # plt.subplot(2, 2, 1), plt.imshow(limg, cmap='gray')
    # plt.subplot(2, 2, 2), plt.imshow(th, cmap='gray')
    # plt.subplot(2, 2, 3), plt.imshow(dilated, cmap='gray')
    # plt.subplot(2, 2, 4), plt.imshow(cropped, cmap='gray')
    # plt.show()

    return cropped


files = glob.glob("/home/kary/Pic/*.bmp")
print(files)
for idx, file in enumerate(files):
    _img = cv.imread(file, 0)
    cropped = cropper(_img)
    fname = "pic_"+str(idx)+".bmp"
    cv.imwrite(fname, cropped)
