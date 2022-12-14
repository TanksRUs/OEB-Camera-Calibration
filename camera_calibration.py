import numpy as np
import cv2 as cv
import glob
from tkinter import filedialog
import csv

# ------CHECKERBOARD PROPERTIES------
ROWS = 10 - 1 # start indexing at 0
COLUMNS = 15 - 1
GRID_SIZE = 1 # in [mm], 1 for unknown/dimensionless (NOTE: literally does not affect calibration parameters)
RESOLUTION_SCALE = 1 # to divide focal lengths/principal points by
# -----------------------------------
IMAGE_EXTS = ['jpg', 'jpeg', 'png'] # valid image extensions
csv_path = filedialog.asksaveasfilename(title='Select or save calibration CSV:', initialfile='calibration.csv', filetypes=(('CSV', '*.csv'), ('All files', '*.*')))

img_path = filedialog.askdirectory(title='Select images folder:')
if img_path == '':
    print('No folder selected!')
    quit()

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001) # (COUNT, MAX_ITER, EPS)
chessboard_flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE

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
    ret, corners = cv.findChessboardCorners(gray, (COLUMNS, ROWS), chessboard_flags)
    # If found, add object points, image points (after refining them)
    print('{}, {}'.format(ret, fname))
    if ret == True:
        objpoints.append(objp)
        corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners TODO: uncomment next 4 lines to show each checkerboard successfully detected
        # cv.drawChessboardCorners(img, (COLUMNS,ROWS), corners, ret)
        # cv.imshow('Corners Detected (Press any key to continue)', img)
        # cv.waitKey(0)
# cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs  = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# validate distortion coeffs by showing last image undistorted
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
x, y, w, h = roi
dst = cv.resize(dst[y:y+h, x:x+w], (1024, 768))
cv.imshow('Undistorted Image (Press any key to continue)', dst)
cv.waitKey(0)
scaled_mtx = mtx / RESOLUTION_SCALE
scaled_mtx[2, 2] = 1.0

with open(csv_path, 'a', newline='') as f:
    writer = csv.writer(f)
    f.write(img_path.rsplit('/',1)[1] + '\n') # name of dataset/calibration parameters
    f.write('RMS Re-projection Error:\n{}\n'.format(ret))
    f.write('Camera Matrix:\n')
    writer.writerows(scaled_mtx)
    f.write('Distortion Coefficients:\nk1,k2,p1,p2,k3\n')
    writer.writerow(dist[0])
    f.write('\n')

print(ret)
print(mtx)
print(dist)
