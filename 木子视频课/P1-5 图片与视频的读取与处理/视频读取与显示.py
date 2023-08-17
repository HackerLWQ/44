import cv2

import matplotlib.pyplot as plt
import numpy as np

vc = cv2.VideoCapture('test.mp4')

while True:
    ret, frame = vc.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Video', gray)

    if cv2.waitKey(100) & 0xFF == 27:
        break

vc.release()
cv2.destroyAllWindows()