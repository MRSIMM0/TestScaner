import cv2
import numpy as np


def isRect(contour, number):
    peri = cv2.arcLength(contour, True)
    aprox = cv2.approxPolyDP(contour, number*peri, True)
    return aprox


def getTotalContourArea(contourList):
    total = 0
    for x in contourList:
        total = total + cv2.contourArea(x)
    return total


def reorder(points):

    points = points.reshape((4, 2))

    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = points.sum(1)

    myPointsNew[0] = points[np.argmin(add)]
    myPointsNew[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    myPointsNew[1] = points[np.argmin(diff)]
    myPointsNew[2] = points[np.argmax(diff)]
    return myPointsNew


def areaFilter(minArea, inputImage):
    componentsNumber, labeledImage, componentStats, componentCentroids = cv2.connectedComponentsWithStats(
        inputImage, connectivity=4)

    # Get the indices/labels of the remaining components based on the area stat
    # (skip the background component at index 0)
    remainingComponentLabels = [i for i in range(
        1, componentsNumber) if componentStats[i][4] >= minArea]

    # Filter the labeled pixels based on the remaining labels,
    # assign pixel intensity to 255 (uint8) for the remaining pixels
    filteredImage = np.where(
        np.isin(labeledImage, remainingComponentLabels) == True, 255, 0).astype('uint8')

    return filteredImage
