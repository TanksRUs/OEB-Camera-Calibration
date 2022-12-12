import cv2 as cv
import numpy as np
from collections import OrderedDict
from tkinter import filedialog


def read_configs(cameraConfigs_path, masks_path):
    num_cams = 0
    cameras = []
    masks = []
    corners = []
    sizes = []
    with open(cameraConfigs_path, 'r', newline='') as f:
        while True:
            line = f.readline()
            if not line:  # check if we're at the end of the file
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
            corners.append((float(f.readline()), float(f.readline())))
            sizes.append((float(f.readline()), float(f.readline())))

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

    ret_val = {
        'cameras': cameras,
        'corners': corners,
        'sizes': sizes,
        'masks': masks
    }

    # return cameras, corners, sizes, masks
    return ret_val


def stitch_frame(all_imgs, camera_configs):
    cameras = camera_configs['cameras']
    corners = camera_configs['corners']
    sizes = camera_configs['sizes']
    masks = camera_configs['masks']

    ba = cv.detail_BundleAdjusterReproj()  # --ba reproj
    ba_refine_mask = 'xxx_x'  # --ba_refine_mask xxx_x
    warp_type = 'plane'  # --warp plane

    # ----- default values -----
    blend_strength = 5
    blend_type = 'multiband'
    compose_megapix = -1
    conf_thresh = 0.5
    estimator = 'homography'
    expos_comp = 'gain_blocks'
    expos_comp_block_size = 32
    expos_comp_nr_feeds = 1
    expos_comp_type = cv.detail.ExposureCompensator_GAIN_BLOCKS
    features = 'orb'
    finder = cv.SIFT_create()
    match_conf = 0.65
    matcher_type = 'homography'
    range_width = -1
    seam = 'dp_color'
    seam_megapix = 0.1
    seam_work_aspect = 1
    try_cuda = False
    wave_correct = cv.detail.WAVE_CORRECT_HORIZ
    work_megapix = 0.6
    # --------------------------

    sizes = []
    for img in all_imgs:
        sizes.append((img.shape[1], img.shape[0]))

    # is_work_scale_set = False
    # is_seam_scale_set = False
    # is_compose_scale_set = False
    # for full_img in all_imgs:
    #     # full_img_sizes.append((full_img.shape[1], full_img.shape[0]))
    #     if is_work_scale_set is False:
    #         work_scale = min(1.0, np.sqrt(work_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
    #         is_work_scale_set = True
    #     img = cv.resize(src=full_img, dsize=None, fx=work_scale, fy=work_scale, interpolation=cv.INTER_LINEAR_EXACT)
    #     if is_seam_scale_set is False:
    #         seam_scale = min(1.0, np.sqrt(seam_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
    #         seam_work_aspect = seam_scale / work_scale
    #         is_seam_scale_set = True
    #     # img_feat = cv.detail.computeImageFeatures2(finder, img)
    #     # features.append(img_feat)
    #     img = cv.resize(src=full_img, dsize=None, fx=seam_scale, fy=seam_scale, interpolation=cv.INTER_LINEAR_EXACT)
    #     # images.append(img)

    focals = []
    for cam in cameras:
        focals.append(cam.focal)
    focals.sort()
    if len(focals) % 2 == 1:
        warped_image_scale = focals[len(focals) // 2]
    else:
        warped_image_scale = (focals[len(focals) // 2] + focals[len(focals) // 2 - 1]) / 2

    for i in range(0, len(all_imgs)):
        img = all_imgs[i]
        K = cameras[i].K().astype(np.float32)
        mask_warped = masks[i]
        seam_scale = min(1.0, np.sqrt(seam_megapix * 1e6 / (img.shape[0] * img.shape[1])))
        seam_work_aspect = seam_scale / work_scale
        warper = cv.PyRotationWarper(warp_type, warped_image_scale * seam_work_aspect)  # warper could be nullptr?

        # prob issues with scaling image
        corner, image_warped = warper.warp(img, K, cameras[i].R, cv.INTER_LINEAR,
                                           cv.BORDER_REFLECT)  # TODO: cv2.error: OpenCV(4.6.0) D:\a\opencv-python\opencv-python\opencv\modules\stitching\src\warpers.cpp:359: error: (-215:Assertion failed) H.size() == Size(3, 3) && H.type() == CV_32F in function 'cv::detail::AffineWarper::getRTfromHomogeneous'
        image_warped_s = image_warped.astype(np.int16)
        blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
        dst_sz = cv.detail.resultRoi(corners=corners, sizes=sizes)
        blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * blend_strength / 100
        blender = cv.detail_MultiBandBlender()
        blender.setNumBands((np.log(blend_width) / np.log(2.) - 1.).astype(np.int32))
        blender.prepare(dst_sz)
        blender.feed(cv.UMat(image_warped_s), mask_warped, corners[i])

    result = None
    result_mask = None
    result, result_mask = blender.blend(result, result_mask)
    zoom_x = 600.0 / result.shape[1]
    dst = cv.normalize(src=result, dst=None, alpha=255., norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    dst = cv.resize(dst, dsize=None, fx=zoom_x, fy=zoom_x)

    return dst


def read_videos(all_vids, camera_configs, output_path, fps, vid_size):
    caps = []
    # stitched_frames = []

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    out = cv.VideoWriter(output_path, fourcc, fps, vid_size)

    for vid_path in all_vids:
        caps.append(cv.VideoCapture(vid_path))

    while caps[0].isOpened():
        ret = []
        frames = []
        for cap in caps: # read through all videos
            ret_temp, frame_temp = cap.read()
            ret.append(ret_temp)
            frames.append(frame_temp)

        if not all(ret):
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # stitched_frames.append(stitch_frame(frames, camera_configs))
        out.write(stitch_frame(frames, camera_configs))

    for cap in caps:
        cap.release()
    out.release()

    # return stitched_frames


def main():
    cameraConfigs_path = filedialog.askopenfilename(title='Select camera configs:', initialfile='cameraConfigs.cfg', filetypes=(('cfg','*.cfg'),('All files','*.*')))
    masks_path = filedialog.askopenfilename(title='Select masks:', initialfile='masks.yml', filetypes=(('yml','*.yml'),('All files','*.*')))
    all_vids = filedialog.askopenfilenames(title='Select videos to stitch:')
    camera_configs = read_configs(cameraConfigs_path, masks_path)
    output_vid = filedialog.asksaveasfilename(title='Save stitched video:', initialfile='stitched.avi', filetypes=(('avi','*.avi'),('All files','*.*')))
    fps = 30
    vid_size = (720, 1280)
    read_videos(all_vids, camera_configs, output_vid, fps, vid_size)


if __name__ == '__main__':
    main()
