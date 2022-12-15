import numpy as np
import cv2 as cv
import glob
from tkinter import filedialog
import csv

# ------CHECKERBOARD PROPERTIES------
ROWS = 10 - 1 # start indexing at 0
COLUMNS = 15 - 1
GRID_SIZE = 1 # in [mm], 1 for unknown/dimensionless
NUM_SAMPLES = 4 # for de-noising
# -----------------------------------
IMAGE_EXTS = ['jpg', 'jpeg', 'png'] # valid image extensions
CSV_PATH = filedialog.asksaveasfilename(title='Select or save calibration CSV:', initialfile='calibration.csv', filetypes=(('CSV','*.csv'),('All files','*.*')))

img_path = filedialog.askdirectory(title='Select images folder:')
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

num_views = int(len(images) / NUM_SAMPLES)

for i in range(0, num_views):
    img = None

    for sample in range (0, NUM_SAMPLES): # average out samples to remove noise
        fname = images[i * NUM_SAMPLES + sample]
        average = cv.imread(fname)
        if img is None:
            img = average / NUM_SAMPLES
        else:
            img += average / NUM_SAMPLES

    img = img.astype('uint8')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (COLUMNS, ROWS), None)
    # If found, add object points, image points (after refining them)
    print('{}, {}'.format(ret, fname))
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        # Draw and display the corners
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv.drawChessboardCorners(img, (COLUMNS,ROWS), corners2, ret)
        cv.imshow('Corners Detected (Press any key to continue)', img)
        cv.waitKey(0)
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