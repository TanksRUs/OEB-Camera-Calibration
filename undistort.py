import cv2 as cv
import numpy as np
from tkinter import filedialog

# -----COPY & PASTED FROM CSV-----
mtx = [[856.2565198058854,0.0,523.5020254748135],
[0.0,859.731351927642,373.8238948932486],
[0.0,0.0,1.0]]
mtx = np.array(mtx)

dist = [[-0.2534300800817647,0.20220112243912391,-0.000777285572553943,-0.00034570247103487444,-0.1215798893894797]]
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


