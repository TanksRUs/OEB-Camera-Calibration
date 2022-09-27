import cv2 as cv
import os
from tkinter import filedialog

# ------CHECKERBOARD PROPERTIES------
ROWS = 6
COLUMNS = 9
# -----------------------------------

vid_paths = filedialog.askopenfilenames(initialdir='C:/Users/duanr/Desktop/Camera Calibration/Raw Images/')
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001) # (COUNT, MAX_ITER, EPS)

if vid_paths == '':
    print('No file selected!')
    quit()

for vid_path in vid_paths:
    vid_name = os.path.splitext(os.path.basename(vid_path))[0]
    imgs_path = os.path.dirname(vid_path) + '/{}/Frames/'.format(vid_name)
    checker_path = os.path.dirname(vid_path) + '/{}/Corners/'.format(vid_name)
    os.makedirs(imgs_path)
    os.makedirs(checker_path)
    vidcap = cv.VideoCapture(vid_path)
    success, image = vidcap.read()
    count = 0

    while success:
        img_name = '{}{}_{}.jpg'.format(imgs_path, vid_name, str(count).zfill(5))

        # draw corners and save image for verification
        checker_name = '{}checker_{}_{}.jpg'.format(checker_path, vid_name, str(count).zfill(5))
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (COLUMNS, ROWS), None)
        print('{}, {}'.format(ret, img_name))

        if ret == True: # save video frame if corners are detected
            cv.imwrite(img_name, image)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv.drawChessboardCorners(image, (COLUMNS, ROWS), corners2, ret)
            cv.imwrite(checker_name, image)

        success, image = vidcap.read()
        count += 1