import cv2 as cv
import numpy as np
from tkinter import filedialog

# -----COPY & PASTED FROM CSV-----
mtx = [[878.2559535058177,0.0,513.5845774258821],
[0.0,880.1219017001371,365.1046934779041],
[0.0,0.0,1.0]]
mtx = np.array(mtx)

dist = [[-0.22471543357445162,0.1449300246842861,-0.0016430226203250154,0.0011252955204544151,-0.0781491098440845]]
dist = np.array(dist)
# --------------------------------

img_path = filedialog.askopenfilename(initialdir='C:/Users/duanr/Desktop/Camera Calibration/Raw Images/')
img = cv.imread(img_path)
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
cv.imshow('img', dst)
cv.waitKey(0)


