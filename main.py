import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from flask import Flask, render_template, Response
import cv2
cv2.setNumThreads(1)
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

web = Flask(__name__)
web.config['THREADS_PER_PAGE'] = 1

detector = None
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

labels = [
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
    "Calm Down","Hello","Love","Stand","Thumbs Up","Where"
]

state = {
    "sentence": "",
    "last_char": "",
    "frame_count": 0
}

COOLDOWN_FRAMES = 15

camera = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
if not camera.isOpened():
    raise RuntimeError("❌ Camera could not be opened. Check permissions or camera index.")

@web.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    detector = HandDetector(maxHands=1)
    try:
        while True:
            success, img = camera.read()
            if not success or img is None:
                continue

            imgOutput = img.copy()
            hands, img = detector.findHands(img)

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

                    current_char = labels[index]
                    state['frame_count'] += 1
                    if current_char != state['last_char'] and state['frame_count'] > COOLDOWN_FRAMES:
                        state['sentence']+= current_char
                        state['last_char'] = current_char
                        state['frame_count'] = 0

                    cv2.putText(
                        imgOutput, labels[index],
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_COMPLEX,
                        1.2, (255, 78, 63), 3
                    )

                    cv2.rectangle(
                        imgOutput,
                        (x1, y1),
                        (x2, y2),
                        (70, 252, 255), 4
                    )

                    cv2.rectangle(imgOutput, (10, 430), (630, 480), (0, 0, 0), -1)
                    cv2.putText(
                        imgOutput,
                        state["sentence"],
                        (20, 465),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 255, 255),
                        2
                    )

            ret, buffer = cv2.imencode('.jpg', imgOutput)
            frame = buffer.tobytes()

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
            )
    except Exception as e:
        print("❌ Error in gen_frames:", e)

@web.route('/video_feed')
def video_feed():
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@web.route('/sentence')
def get_sentence():
    return state["sentence"]

if __name__ == '__main__':
    web.run(host="0.0.0.0", port=5000, debug=True, threaded=True, use_reloader=False)