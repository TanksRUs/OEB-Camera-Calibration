import cv2 as cv
import numpy as np
import csv
from tkinter import filedialog
import glob
import os

# -----READ CSV-----
NUM_CAMS = 4
# csv_path = 'C:/Users/duanr/Desktop/Stitching/calibration.csv'
csv_path = filedialog.askopenfilename(title='Select Camera Calibration CSV', initialdir='C:/Users/duanr/Desktop/Stitching/')
mtxs = []
dists = []

with open(csv_path, newline='') as csv_file:
    reader = csv.reader(csv_file)
    for i in range(0, NUM_CAMS):
        for header in range(0, 4):  # skips header rows
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
        try:  # in case there's no line return at the end of the file
            next(reader)
        except StopIteration:
            pass
# --------------------------------

folder_path = filedialog.askdirectory(title='Select Distorted Images Folder', initialdir='C:/Users/duanr/Desktop/Stitching/')
if folder_path == '':
    print('No folder selected!')
    quit()
output_folder = os.path.dirname(folder_path) + '/Undistorted'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

images = glob.glob(folder_path + '/*.png')  # TODO: make sure this matches the image file type
count = 0
command = 'stitching_detailed.py'

for img_path in images: # assuming images are in the same order as the camera order
    print(img_path)
    camera = count % NUM_CAMS
    mtx = mtxs[camera]
    dist = dists[camera]
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    new_path = '{}/{}_undist.png'.format(output_folder, img_name)

    img = cv.imread(img_path)
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    try:
        cv.imwrite(new_path, dst)
    except FileExistsError:
        print('File already exists: ' + img_name)

    command += ' ' + new_path
    # cv.imshow('img', dst)
    # cv.waitKey(0)
    count += 1

command += ' --warp affine --matcher affine --estimator affine --ba affine --wave_correct no --output_folder {}/'.format(os.path.dirname(folder_path))
os.system(command)
# parameters for stitching_detailed:
# --warp affine --matcher affine --estimator affine --ba affine --wave_correct no