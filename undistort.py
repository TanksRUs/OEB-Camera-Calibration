import cv2 as cv
import numpy as np
import csv
from tkinter import filedialog

# -----READ CSV-----
# csv_path = filedialog.askopenfilename(initialdir='C:/Users/duanr/Desktop/Stitching/')
NUM_CAMS = 4
csv_path = 'C:/Users/duanr/Desktop/Stitching/calibration.csv'
mtxs = []
dists = []

with open(csv_path, newline='') as csv_file:
    reader = csv.reader(csv_file)
    for i in range(0, NUM_CAMS):
        for header in range (0, 4): # skips header rows
            next(reader)
        mtx = []
        mtx.append([float(x) for x in next(reader)])
        mtx.append([float(x) for x in next(reader)])
        mtx.append([float(x) for x in next(reader)])
        mtx = np.array(mtx)
        mtxs.append(mtx)
        next(reader)
        next(reader)
        dist = [[float(x) for x in next(reader)]]
        dist = np.array(dist)
        dists.append(dist)
        try: # in case there's no line return at the end of the file
            next(reader)
        except StopIteration:
            pass
# --------------------------------

folder_path = filedialog.askdirectory(initialdir='C:/Users/duanr/Desktop/Stitching/')
img = cv.imread(img_path)
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
cv.imshow('img', dst)
cv.waitKey(0)


