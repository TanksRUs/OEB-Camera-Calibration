import cv2 as cv
import numpy as np
import csv
from tkinter import filedialog
import glob
import os

IMAGE_EXTS = ['jpg', 'jpeg', 'png'] # valid image extensions
# -----READ CSV-----
NUM_CAMS = 4
csv_path = filedialog.askopenfilename(title='Select Camera Calibration CSV', initialfile='calibration.csv')
var_Ks = []
var_Ds = []

with open(csv_path, newline='') as csv_file:
    reader = csv.reader(csv_file)
    for i in range(0, NUM_CAMS):
        for header in range(0, 4):  # skips header rows
            next(reader)
        var_K = []
        var_K.append([float(x) for x in next(reader)])
        var_K.append([float(x) for x in next(reader)])
        var_K.append([float(x) for x in next(reader)])
        var_K = np.array(var_K)
        var_Ks.append(var_K)
        next(reader)
        next(reader)
        var_D = [[float(x) for x in next(reader)]]
        var_D = np.array(var_D)
        var_Ds.append(var_D)
        try:  # in case there's no line return at the end of the file
            next(reader)
        except StopIteration:
            pass
# --------------------------------

folder_path = filedialog.askdirectory(title='Select original images folder:')
if folder_path == '':
    print('No folder selected!')
    quit()
output_folder = os.path.dirname(folder_path) + '/Undistorted'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for extension in IMAGE_EXTS:
    images = glob.glob('{}/*.{}'.format(folder_path, extension))
    if images: # if there are files with the specified extension
        break
count = 0

for img_path in images: # assuming images are in the same order as the camera order
    print(img_path)
    camera = count % NUM_CAMS
    var_K = var_Ks[camera]
    var_D = var_Ds[camera]
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    new_path = '{}/{}_undist.png'.format(output_folder, img_name)

    img = cv.imread(img_path)
    h, w = img.shape[:2]
    map1, map2 = cv.fisheye.initUndistortRectifyMap(var_K, var_D, np.eye(3), var_K, (w, h), cv.CV_16SC2)
    dst = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)

    try:
        cv.imwrite(new_path, dst)
    except FileExistsError:
        print('File already exists: ' + img_name)

    # cv.imshow('img', dst)
    # cv.waitKey(0)
    count += 1
