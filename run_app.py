import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import cv2
cv2.setNumThreads(1)

from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from collections import deque
pred_buffer = deque(maxlen=10)

cap = cv2.VideoCapture(0)

for _ in range(10):
    cap.read()

detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

with open("Model/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

sentence = ""
last_char = ""
offset = 20
imgSize = 300

while True:
    success, img = cap.read()
    if not success:
        break

    imgOutput = img.copy()
    hands, img = detector.findHands(img, draw=False)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        hImg, wImg, _ = img.shape
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(wImg, x + w + offset)
        y2 = min(hImg, y + h + offset)

        imgCrop = img[y1:y2, x1:x2]
        if imgCrop.size != 0:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            confidence = prediction[index]

            cv2.putText(
                imgOutput,
                f"conf={confidence:.2f}",
                (x1, y1 - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            valid_char = True

            if confidence < 0.50:
                valid_char = False
            else:
                char = labels[index]
                cv2.putText(
                    imgOutput,
                    f"raw={char}",
                    (x1, y1 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )
                if not char.isalpha():
                    valid_char = False

            if valid_char:
                pred_buffer.append(char)

                most_common = max(set(pred_buffer), key=pred_buffer.count)

                if pred_buffer.count(most_common) >= 5 and most_common != last_char:
                    sentence += most_common
                    last_char = most_common
                    pred_buffer.clear()

                display_char = most_common if pred_buffer.count(most_common) >= 7 else ""
                if display_char:
                    cv2.putText(
                        imgOutput,
                        display_char,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1.2,
                        (0, 255, 0),
                        3
                    )

    cv2.rectangle(imgOutput, (10, 430), (630, 480), (0,0,0), -1)
    cv2.putText(imgOutput, sentence, (20, 465),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Hand Sign Detector", imgOutput)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()