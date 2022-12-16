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
calibration_flags = cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv.fisheye.CALIB_FIX_SKEW # removed cv.fisheye.CALIB_CHECK_COND
chessboard_flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE

objp = np.zeros((1, ROWS * COLUMNS, 3), np.float32)
objp[0, :, :2] = np.mgrid[0:COLUMNS, 0:ROWS].T.reshape(-1, 2) # point locations (cur. unknown dims)
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
        # Draw and display the corners
        # cv.drawChessboardCorners(img, (COLUMNS,ROWS), corners, ret)
        # cv.imshow('Chessboard Corners (Press any key to continue)', img)
        # cv.waitKey(0)
# cv.destroyAllWindows()

N_OK = len(objpoints)
var_K = np.zeros((3, 3))
var_D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

ret, var_K, var_D, rvecs, tvecs = \
    cv.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        var_K,
        var_D,
        rvecs,
        tvecs,
        calibration_flags,
        criteria
    )

with open(csv_path, 'a', newline='') as f:
    writer = csv.writer(f)
    f.write(img_path.rsplit('/',1)[1] + '\n') # name of dataset/calibration parameters
    # f.write('RMS Re-projection Error:\n{}\n'.format(ret)) # no idea what ret represents for fisheye
    f.write('Camera Matrix:\n')
    writer.writerows(var_K)
    f.write('Distortion Coefficients:\nk1,k2,k3,k4\n')
    writer.writerow(var_D.reshape(1,4)[0])
    f.write('\n')

print(ret)
print(var_K)
print(var_D)
