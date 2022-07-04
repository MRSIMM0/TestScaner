import cv2
import utils
import numpy as np
import math
import string
path = "3.png"

img = cv2.imread(path)


def getWidth(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return w


def getY(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return (y + h)/2


def getAnswer(contour, width):

    x, y, w, h = cv2.boundingRect(contour)
    x = (x + w)/2
    y = (y + h)/2
    print(string.ascii_uppercase[math.floor((x*6)/width)])


def splitImage(warppedPart, height, width):

    h, w, channels = warppedPart.shape

    area = h/19 * w/6

    warppedPart = cv2.resize(warppedPart, (w*2, h*2))
    im = warppedPart.copy()
    # kernel = np.ones((5, 5), np.uint8)

    # warppedPart = cv2.dilate(warppedPart, kernel, iterations=3)

    # for x in range(3):
    #     warppedPart = cv2.detailEnhance(warppedPart, sigma_s=10, sigma_r=0.15)
    # warppedPart = cv2.erode(warppedPart, kernel, iterations=1)

    imgGray = cv2.cvtColor(warppedPart, cv2.COLOR_BGR2GRAY)

    binaryThresh = 90
    _, binaryImage = cv2.threshold(
        imgGray, binaryThresh, 255, cv2.THRESH_BINARY)
    # minArea = 1000
    # binaryImage = utils.areaFilter(minArea, binaryImage)

    binaryImage = utils.areaFilter(10000, binaryImage)

    kernelSize = 3
    # Set morph operation iterations:
    opIterations = 2
    # Get the structuring element:
    morphKernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernelSize, kernelSize))
    # Perform closing:
    imgGray = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE,
                               morphKernel, None, None, opIterations, cv2.BORDER_REFLECT101)

    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)

    imgCanny = cv2.Canny(imgBlur, 200, 490)

    contours, hierarchy = cv2.findContours(
        imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = [c for c in contours if cv2.contourArea(
        c) > 3000]

    cv2.drawContours(im, contours, -1, (0, 255, 0), 3)

    contours = sorted(contours, key=lambda c: getY(c))

    for x in contours:
        getAnswer(x, w)

    im = cv2.resize(im, (w, h))

    cv2.imshow(str(im), im)


def evaluate(img):

    alpha = 1.3

    beta = -5

    height, width, channels = img.shape

    img = cv2.resize(img, (width*4, height*4))

    img2 = img.copy()

    imgRes = img.copy()

    img = cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)

    kernel = np.ones((5, 5), np.uint8)

    img = cv2.erode(img, kernel, iterations=1)

    img = cv2.dilate(img, kernel, iterations=1)

    for x in range(3):
        img = cv2.detailEnhance(img, sigma_s=10, sigma_r=0.15)
    img = cv2.erode(img, kernel, iterations=1)

    # for x in range(1):
    #     img = cv2.stylization(img, sigma_s=60, sigma_r=0.02)
    # for x in range(2):
    #     img = cv2.detailEnhance(img, sigma_s=30, sigma_r=0.15)

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    binaryThresh = 160
    _, binaryImage = cv2.threshold(
        imgGray, binaryThresh, 255, cv2.THRESH_BINARY)
    # minArea = 1000
    # binaryImage = utils.areaFilter(minArea, binaryImage)

    kernelSize = 3
    # Set morph operation iterations:
    opIterations = 2
    # Get the structuring element:
    morphKernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (kernelSize, kernelSize))
    # Perform closing:
    imgGray = cv2.morphologyEx(binaryImage, cv2.MORPH_CLOSE,
                               morphKernel, None, None, opIterations, cv2.BORDER_REFLECT101)

    imgBlur = cv2.GaussianBlur(imgGray, (5, 5), 0)

    imgCanny = cv2.Canny(imgBlur, 140, 150)
    contours, hierarchy = cv2.findContours(
        imgCanny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted([utils.isRect(x, 0.03)
                       for x in contours if len(utils.isRect(x, 0.03)) == 4 and cv2.contourArea(utils.isRect(x, 0.03)) > 8000 and cv2.contourArea(utils.isRect(x, 0.03)) < 50000 and getWidth(x) > 150], key=lambda c: cv2.contourArea(c), reverse=True)

    contours = contours[-1]
    contours = utils.reorder(contours)

    x, y, w, h = cv2.boundingRect(contours)

    contours[0][0][0] = contours[0][0][0]-10

    contours[1][0][0] = contours[1][0][0] + 10

    contours[0][0][1] = contours[0][0][1] + h

    contours[1][0][1] = contours[1][0][1] + h

    contours[2][0][0] = contours[0][0][0]

    contours[3][0][0] = contours[1][0][0]

    contours[2][0][1] = height*4 - 10
    contours[3][0][1] = height*4 - 10

    cv2.drawContours(imgRes, contours, -1, (0, 255, 0), 9)

    offset = 5

    contours[0][0][0] = contours[0][0][0] - offset
    contours[0][0][1] = contours[0][0][1] - offset
    contours[1][0][0] = contours[1][0][0] + offset
    contours[1][0][1] = contours[1][0][1] - offset
    contours[2][0][0] = contours[2][0][0] - offset
    contours[2][0][1] = contours[2][0][1] + offset
    contours[3][0][0] = contours[3][0][0] + offset
    contours[3][0][1] = contours[3][0][1] + offset

    pt1 = np.float32(contours)

    x, y, w, he = cv2.boundingRect(contours)

    pt2 = np.array([[0, 0], [w, 0], [0, he],
                    [w, he]], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(pt1, pt2)

    warpedImage = cv2.warpPerspective(img2, matrix, (w, he))

    splitImage(warpedImage, h, w)
