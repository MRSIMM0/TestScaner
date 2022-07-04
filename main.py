import cv2
import utils
import numpy as np
import marking

path = "1.jpg"
imgCont2 = None
vid = cv2.VideoCapture(2)
maxCountoursArea = []

width = 0
height = 0


def xcord(contour):
    x, y, w, h = cv2.boundingRect(contour)
    return x


while True:

    ret, frame = vid.read()

    img = frame

    imgCont = img.copy()
    imgCont2 = img.copy()

    height, width, channels = img.shape

    alpha = 1.5

    beta = 0

    img = cv2.addWeighted(img, alpha, np.zeros(img.shape, img.dtype), 0, beta)

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    imgBlur = cv2.GaussianBlur(imgGray, (1, 1), 0.7)
    imgCanny = cv2.Canny(imgBlur, 40, 60)
    contours, hierarchy = cv2.findContours(
        imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted([utils.isRect(x, 0.036) for x in contours if len(utils.isRect(x, 0.036)) == 4 and cv2.contourArea(utils.isRect(x, 0.036)) > 20000],
                      key=lambda c: cv2.contourArea(c), reverse=True)[:2]

    contours = sorted(contours, key=lambda c: xcord(c), reverse=False)

    if(len(contours) == 2 and len(maxCountoursArea) < 6):
        maxCountoursArea.append(contours)
    cv2.drawContours(imgCont, contours, -1, (0, 255, 0), 3)
    cv2.imshow("Przeszukiwacz", imgCont)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice

    # print(cv2.contourArea(maxCountoursArea[-1][0]))

    if (cv2.waitKey(1) & 0xFF == ord('q') or len(maxCountoursArea) == 3):
        break
cv2.destroyAllWindows()
img = imgCont2.copy()
bestChice = (0, 0)

for i in range(len(maxCountoursArea)-1):
    if(bestChice[1] < utils.getTotalContourArea(maxCountoursArea[i])):
        bestChice = (maxCountoursArea[i], utils.getTotalContourArea(
            maxCountoursArea[i]))

warrpedImages = []
for x in bestChice[0]:
    x = utils.reorder(x)

    offset = 5

    x[0][0][0] = x[0][0][0] - offset
    x[0][0][1] = x[0][0][1] - offset
    x[1][0][0] = x[1][0][0] + offset
    x[1][0][1] = x[1][0][1] - offset
    x[2][0][0] = x[2][0][0] - offset
    x[2][0][1] = x[2][0][1] + offset
    x[3][0][0] = x[3][0][0] + offset
    x[3][0][1] = x[3][0][1] + offset

    pt1 = np.float32(x)

    x, y, w, h = cv2.boundingRect(x)

    pt2 = np.array([[0, 0], [w, 0], [0, h],
                    [w, h]], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(pt1, pt2)

    warrpedImages.append(cv2.warpPerspective(img, matrix, (w, h)))

    process = []

for w in warrpedImages:

    marking.evaluate(w)
    #     process.append(threading.Thread(
    #         target=, args=(w,), daemon=True))

    # for p in process:
    #     p.start()

    # for p in process:
    #     p.join()


cv2.waitKey(0)
# Destroy all the windows
