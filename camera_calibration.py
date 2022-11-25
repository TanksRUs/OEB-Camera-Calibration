import numpy as np
import cv2 as cv
import glob
from tkinter import filedialog
import csv

# ------CHECKERBOARD PROPERTIES------
ROWS = 9 - 1 # start indexing at 0
COLUMNS = 7 - 1
GRID_SIZE = 1 # in [mm], 1 for unknown/dimensionless
# -----------------------------------
IMAGE_EXTS = ['jpg', 'jpeg', 'png'] # valid image extensions
CSV_PATH = filedialog.askdirectory()
CSV_PATH += '/calibration.csv'
# CSV_PATH = 'C:/Users/duanr/Desktop/Camera Calibration/calibration.csv'

img_path = filedialog.askdirectory()
if img_path == '':
    print('No folder selected!')
    quit()

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001) # (COUNT, MAX_ITER, EPS)

objp = np.zeros((ROWS * COLUMNS, 3), np.float32)
objp[:, :2] = np.mgrid[0:COLUMNS, 0:ROWS].T.reshape(-1, 2) # point locations (cur. unknown dims)
objp *= GRID_SIZE

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

for extension in IMAGE_EXTS:
    images = glob.glob('{}/*.{}'.format(img_path, extension))
    if images: # if there are files with the specified extension
        break

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (COLUMNS, ROWS), None)
    # If found, add object points, image points (after refining them)
    print('{}, {}'.format(ret, fname))
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        # Draw and display the corners
        # corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # cv.drawChessboardCorners(img, (COLUMNS,ROWS), corners2, ret)
        # cv.imshow('img', img)
        # cv.waitKey(0)
# cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs  = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

with open(CSV_PATH, 'a', newline='') as f:
    writer = csv.writer(f)
    f.write(img_path.rsplit('/',1)[1] + '\n') # name of dataset/calibration parameters
    f.write('RMS Re-projection Error:\n{}\n'.format(ret))
    f.write('Camera Matrix:\n')
    writer.writerows(mtx)
    f.write('Distortion Coefficients:\nk1,k2,p1,p2,k3\n')
    writer.writerow(dist[0])
    f.write('\n')

print(ret)
print(mtx)
print(dist)
# print(imgpoints)