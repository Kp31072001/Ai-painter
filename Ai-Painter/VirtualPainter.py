from typing import overload
import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm


#############################
brushThickness = 15
ErasserThickness = 50
#############################

folderPath = "header"
myList = os.listdir(folderPath)
myList[0] = myList[4]
myList[4] = myList[1]
myList[1] = myList[2]
myList[2] = myList[3]
myList[3] = myList[4]
del myList[4]
print(myList)

overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))

header = overlayList[0]
drawColor = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = htm.handDetector(detectionCon=0.85)

xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)

while True:
    # steps
    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img, 1)

    # 2. Find Hand Landmark
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # print(lmList)

        # tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. Check which finger are up
        fingers = detector.fingersUp()
        # print(fingers)

        # 4. If selection mode - Two finger sare up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0

            print("Selection Mode")
            # checking for click
            if y1 < 98:
                if 250 < x1 < 450:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 550 < x1 < 750:
                    header = overlayList[1]
                    drawColor = (255, 0, 20)
                elif 800 < x1 < 950:
                    header = overlayList[2]
                    drawColor = (0, 255, 0)
                elif 1050 < x1 < 1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1-25), (x2, y2+25),
                          drawColor, cv2.FILLED)

        # 5. If Drawing mode - only Idex finger up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, ErasserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1),
                         drawColor, ErasserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1),
                         drawColor, brushThickness)

            xp, yp = x1, y1

    # drawing on the real img
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    h, w, c = header.shape
    img[0:h, 0:w] = header

    # there will be a transperancy
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)

    cv2.imshow("Image", img)
    # cv2.imshow("Camvas", imgCanvas)
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break
