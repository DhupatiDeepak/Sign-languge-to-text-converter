import cv2
import time
import mediapipe as mp

handsUtil = mp.solutions.hands
handsDraw = mp.solutions.drawing_utils

hands = handsUtil.Hands(static_image_mode=False, max_num_hands=3, min_detection_confidence=0.6, min_tracking_confidence=0.6)

cam = cv2.VideoCapture(0)

fingerTipList = [4, 8, 12, 16, 20]

gestures = {
    "A": [1, 0, 0, 0, 0],
    "B": [0, 1, 0, 0, 0],
    "C": [0, 0, 1, 0, 0],
    "D": [-1, 0, 0, 0, 1],
    "E": [0, 0, 0, 0, 1],
    "F": [1, 1, 0, 0, 0],
    "G": [0, 1, 1, 0, 0],
    "H": [0, 0, 1, 1, 0],
    "I": [0, 0, 0, 1, 1],
    "J": [1, 1, 1, 0, 0],
    "K": [0, 1, 1, 1, 0],
    "L": [0, 0, 1, 1, 1],
    "M": [1, 1, 1, 1, 0],
    "N": [0, 1, 1, 1, 1],
    "O": [0, 0, 0, 0, 0],
    "P": [1, 0, 0, 0, 1],
    "Q": [-1, 1, 0, 0, 0],
    "R": [1, 0, 1, 0, 1],
    "S": [1, 0, 1, 1, 1],
    "T": [1, 1, 1, 0, 1],
    "U": [0, 1, 1, 0, 1],
    "V": [-1, 1, 1, 0, 0],
    "W": [-1, 1, 0, 0, 1],
    "X": [0, 1, 0, 0, 1],
    "Y": [1, 1, 0, 0, 1],
    "Z": [1, 1, 1, 1, 1]
}

current_word = ""
recognized_word = ""
last_gesture_time = time.time()

currentTime = 0
previousTime = 0

while True:
    success, frame = cam.read()
    frameresize = cv2.resize(frame, (720, 480))
    frameflip = cv2.flip(frameresize, 1)
    framergb = cv2.cvtColor(frameflip, cv2.COLOR_BGR2RGB)

    results = hands.process(framergb)

    HandList = []

    if results.multi_hand_landmarks:
        for handlandmark in results.multi_hand_landmarks:
            handsDraw.draw_landmarks(frameflip, handlandmark, handsUtil.HAND_CONNECTIONS)
            for pointid, landmark in enumerate(handlandmark.landmark):
                h, w, c = framergb.shape
                x, y = int(landmark.x * w), int(landmark.y * h)
                HandList.append([pointid, x, y])

    if len(HandList) == 21:
        cv2.putText(frameflip, "Hand Detected", (10, 55), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 55, 55), 1)
        fingerPos = []

        # THUMB
        if HandList[fingerTipList[0]][1] < HandList[fingerTipList[4]][1]:
            cv2.putText(frameflip, "Palm - Correct", (10, 85), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 55, 55), 1)
            if HandList[fingerTipList[0]][1] > HandList[fingerTipList[0] - 2][1]:
                fingerPos.append(0)
            else:
                fingerPos.append(1)
        else:
            cv2.putText(frameflip, "Palm - Invert", (10, 85), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 55, 55), 1)
            if HandList[fingerTipList[0]][1] < HandList[fingerTipList[0] - 2][1]:
                fingerPos.append(0)
            else:
                fingerPos.append(-1)

        # Other finger
        for id in range(1, 5):
            if HandList[fingerTipList[id]][2] > HandList[fingerTipList[id] - 2][2]:
                fingerPos.append(0)
            else:
                fingerPos.append(1)

        print(fingerPos)

        for gesture, value in gestures.items():
            if fingerPos == value:
                if time.time() - last_gesture_time >= 3:
                    recognized_word = gesture
                    current_word += recognized_word
                    last_gesture_time = time.time()
                break
        else:
            recognized_word = ""

        if recognized_word:
            text_to_display = f"TEXT: Current - {current_word}, Recognized - {recognized_word}"
            cv2.putText(frameflip, text_to_display, (10, 115), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 55, 255), 1)
        else:
            cv2.putText(frameflip, "TEXT: IDLE (UNKNOWN GESTURE)", (10, 115), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 55, 255), 1)

    currentTime = time.time()
    fps = int(1 / (currentTime - previousTime))
    previousTime = currentTime

    cv2.putText(frameflip, str(fps), (10, 25), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 55, 55), 1)
    cv2.imshow("Frame Window", frameflip)
    cv2.waitKey(1)
