import cv2 as cv
import numpy as np
from tkinter import filedialog
import os
import glob

# cameraConfigs_path = filedialog.askopenfilename(title='cameraConfigs.cfg', initialdir='C:\\Users\\duanr\\Desktop\\Stitching')
# masks_path = filedialog.askopenfilename(title='masks.yml', initialdir='C:\\Users\\duanr\\Desktop\\Stitching')
# images_folder_path = filedialog.askdirectory(title='Images to stitch:', initialdir='C:\\Users\\duanr\\Desktop\\Stitching')
cameraConfigs_path = 'C:\\Users\\duanr\\Desktop\\Stitching\\cameraConfigs.cfg'
masks_path = 'C:\\Users\\duanr\\Desktop\\Stitching\\masks.yml'
images_folder_path = 'C:\\Users\\duanr\\Desktop\\Stitching\\Undistorted\\'
image_paths = glob.glob(images_folder_path + '/*.png')

num_cams = 0
cameras = []
masks = []

with open(cameraConfigs_path, 'r', newline='') as f:
    # reader = csv.reader(f)
    while True:
        line = f.readline()
        if not line: # check if we're at the end of the file
            break
        aspect = float(line)
        focal = float(f.readline())
        ppx = float(f.readline())
        ppy = float(f.readline())
        R = []
        for i in range(0, 3):
            R.append([float(f.readline()), float(f.readline()), float(f.readline())])
        R = np.array(R)
        t = [[float(f.readline())], [float(f.readline())], [float(f.readline())]]
        t = np.array(t)
        corners = (float(f.readline()), float(f.readline()))

        camera = cv.detail.CameraParams()
        camera.aspect = aspect
        camera.focal = focal
        camera.ppx = ppx
        camera.ppy = ppy
        camera.R = R
        camera.t = t
        cameras.append(camera)
        num_cams += 1

mask_file = cv.FileStorage(masks_path, cv.FileStorage_READ)
for view in range(0, num_cams):
    mask_count = 'mask{}'.format(view)
    masks.append(mask_file.getNode(mask_count).mat())
mask_file.release()

warp_type = 'affine'
warped_image_scale = 1
seam_work_aspect = 1 # not exactly the same as stitching_detailed.py but shouldn't affect end result?
warper = cv.PyRotationWarper(warp_type, warped_image_scale * seam_work_aspect)  # warper could be nullptr?
blend_strength = 5

sizes = []
for name in image_paths:
    img = cv.imread(name)
    sizes.append((img.shape[1], img.shape[0]))

for idx, name in enumerate(image_paths):
    img = cv.imread(name)
    K = cameras[idx].K().astype(np.float32)
    mask_warped = masks[idx]

    # prob issues with scaling image
    corner, image_warped = warper.warp(img, K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT) # TODO: cv2.error: OpenCV(4.6.0) D:\a\opencv-python\opencv-python\opencv\modules\stitching\src\warpers.cpp:359: error: (-215:Assertion failed) H.size() == Size(3, 3) && H.type() == CV_32F in function 'cv::detail::AffineWarper::getRTfromHomogeneous'
    image_warped_s = image_warped.astype(np.int16)
    blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
    dst_sz = cv.detail.resultRoi(corners=corners, sizes=sizes)
    blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * blend_strength / 100
    blender = cv.detail_MultiBandBlender()
    blender.setNumBands((np.log(blend_width) / np.log(2.) - 1.).astype(np.int32))
    blender.prepare(dst_sz)
    blender.feed(cv.UMat(image_warped_s), mask_warped, corners[idx])

result = None
result_mask = None
result, result_mask = blender.blend(result, result_mask)
zoom_x = 600.0 / result.shape[1]
dst = cv.normalize(src=result, dst=None, alpha=255., norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
dst = cv.resize(dst, dsize=None, fx=zoom_x, fy=zoom_x)
cv.imshow('Stitched Image', dst)
# K = cameras[i].K().astype(np.float32)