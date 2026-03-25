import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from flask import Flask, render_template, Response, jsonify, request
import cv2
cv2.setNumThreads(1)
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
from collections import deque

web = Flask(__name__)
web.config['THREADS_PER_PAGE'] = 1

classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Read labels from file to stay in sync with the model
with open("Model/labels.txt", "r") as f:
    labels = [line.strip().split(" ", 1)[1] for line in f.readlines() if line.strip()]

# Only accept predictions for classes that actually exist in the DATA folder
valid_labels = set()
data_dir = os.path.join(os.path.dirname(__file__), "DATA")
if os.path.isdir(data_dir):
    valid_labels = {d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))}

offset = 20
imgSize = 300

# Prediction buffer for stable detection (ported from run_app.py)
pred_buffer = deque(maxlen=15)

state = {
    "sentence": "",
    "last_char": "",
    "current_char": "",
    "current_confidence": 0.0,
    "camera_on": True,
}

camera = None

def init_camera():
    global camera
    try:
        cam = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        if not cam.isOpened():
            # Try without specifying backend
            cam = cv2.VideoCapture(0)
        if cam.isOpened():
            # Warm up camera
            for _ in range(10):
                cam.read()
            camera = cam
            print("✅ Camera initialized successfully.")
        else:
            print("⚠️ Camera could not be opened. The server will start without camera.")
            print("   Grant camera permission and restart, or toggle camera on from the UI.")
            state["camera_on"] = False
    except Exception as e:
        print(f"⚠️ Camera init error: {e}. Starting without camera.")
        state["camera_on"] = False

init_camera()


@web.route('/')
def index():
    return render_template('index.html')


def gen_frames():
    detector = HandDetector(maxHands=1)
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    try:
        while True:
            if not state["camera_on"] or camera is None:
                # Send a blank frame when camera is toggled off or unavailable
                ret, buffer = cv2.imencode('.jpg', blank_frame)
                frame = buffer.tobytes()
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
                )
                cv2.waitKey(30)
                continue

            success, img = camera.read()
            if not success or img is None:
                continue

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
                    state["current_confidence"] = round(float(confidence), 3)

                    valid_char = True
                    if confidence < 0.65:
                        valid_char = False
                    else:
                        char = labels[index]
                        # Only accept chars that exist in the dataset
                        if valid_labels and char not in valid_labels:
                            valid_char = False

                    if valid_char:
                        pred_buffer.append(char)
                        most_common = max(set(pred_buffer), key=pred_buffer.count)

                        # Update current detected char for the frontend
                        if pred_buffer.count(most_common) >= 7:
                            state["current_char"] = most_common

                        # Add to sentence when stable and different from last
                        if pred_buffer.count(most_common) >= 7 and most_common != state["last_char"]:
                            state["sentence"] += most_common
                            state["last_char"] = most_common
                            pred_buffer.clear()

                    # Draw bounding box outline around detected hand
                    cv2.rectangle(
                        imgOutput,
                        (x1, y1), (x2, y2),
                        (70, 252, 255), 3
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


@web.route('/detected_char')
def get_detected_char():
    return jsonify({"char": state["current_char"]})


@web.route('/confidence')
def get_confidence():
    return jsonify({"confidence": state["current_confidence"]})


@web.route('/camera_toggle', methods=['POST'])
def camera_toggle():
    state["camera_on"] = not state["camera_on"]
    return jsonify({"camera_on": state["camera_on"]})


@web.route('/reset_sentence', methods=['POST'])
def reset_sentence():
    state["sentence"] = ""
    state["last_char"] = ""
    state["current_char"] = ""
    state["current_confidence"] = 0.0
    pred_buffer.clear()
    return jsonify({"ok": True})


if __name__ == '__main__':
    web.run(host="0.0.0.0", port=5000, debug=True, threaded=True, use_reloader=False)