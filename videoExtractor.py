import cv2
import os
from tkinter import filedialog

vid_paths = filedialog.askopenfilenames(initialdir='C:/Users/duanr/Desktop/Camera Calibration/Raw Images/')

if vid_paths == '':
    print('No file selected!')
    quit()

for vid_path in vid_paths:
    vid_name = os.path.splitext(os.path.basename(vid_path))[0]
    imgs_path = os.path.dirname(vid_path) + '/{}/'.format(vid_name)
    os.makedirs(imgs_path)
    vidcap = cv2.VideoCapture(vid_path)
    success, image = vidcap.read()
    count = 0

    while success:
        img_name = '{}{}_{}.jpg'.format(imgs_path, vid_name, str(count).zfill(5))
        print(img_name)
        cv2.imwrite(img_name, image)
        success, image = vidcap.read()
        count += 1