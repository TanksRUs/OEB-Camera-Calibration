import cv2 as cv
import numpy as np
from collections import OrderedDict
from tkinter import filedialog
import csv


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
            corners.append((int(f.readline()), int(f.readline())))
            sizes.append((int(f.readline()), int(f.readline())))

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


# modify stitching parameters here
def first_stitch(all_imgs, camera_configs):
    # ----- stitching parameters ----- (copied from video_generate_configs)
    ba = cv.detail_BundleAdjusterReproj()  # --ba reproj
    ba_refine_mask = 'xxx_x'  # --ba_refine_mask xxx_x
    warp_type = 'plane'  # --warp plane
    blend_strength = 5
    blend_type = 'multiband'
    compose_megapix = -1
    conf_thresh = 0.5  # --conf_thresh 0.5
    estimator = 'homography'
    expos_comp = 'gain_blocks'
    expos_comp_block_size = 32
    expos_comp_nr_feeds = 1
    expos_comp_type = cv.detail.ExposureCompensator_GAIN_BLOCKS
    finder = cv.SIFT_create()  # --features sift
    match_conf = 0.65  # --match_conf 0.65
    matcher_type = 'homography'
    range_width = -1
    seam = 'dp_color'
    seam_megapix = 0.1
    seam_work_aspect = 1
    try_cuda = False
    wave_correct = cv.detail.WAVE_CORRECT_HORIZ
    work_megapix = 0.6
    # --------------------------
    cameras = camera_configs['cameras']
    corners = camera_configs['corners']
    sizes = camera_configs['sizes']
    masks_warped = camera_configs['masks']

    # calculate work_scale
    full_img_sizes = []
    is_work_scale_set = False
    is_seam_scale_set = False
    is_compose_scale_set = False
    for full_img in all_imgs:
        full_img_sizes.append((full_img.shape[1], full_img.shape[0]))
        if is_work_scale_set is False:
            work_scale = min(1.0, np.sqrt(work_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
            is_work_scale_set = True
        # img = cv.resize(src=full_img, dsize=None, fx=work_scale, fy=work_scale, interpolation=cv.INTER_LINEAR_EXACT)
        if is_seam_scale_set is False:
            seam_scale = min(1.0, np.sqrt(seam_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
            seam_work_aspect = seam_scale / work_scale
            is_seam_scale_set = True

    compose_scale = 1
    blender = None

    for idx in range(0, len(all_imgs)):
        full_img = all_imgs[idx]
        if not is_compose_scale_set:
            is_compose_scale_set = True
            compose_work_aspect = compose_scale / work_scale

            # calculate warped_image_scale
            focals = []  # the focals saved have been multiplied by compose_work_aspect
            for cam in cameras:
                focals.append(cam.focal / compose_work_aspect)
            focals.sort()
            if len(focals) % 2 == 1:
                warped_image_scale = focals[len(focals) // 2]
            else:
                warped_image_scale = (focals[len(focals) // 2] + focals[len(focals) // 2 - 1]) / 2

            warped_image_scale *= compose_work_aspect
            warper = cv.PyRotationWarper(warp_type, warped_image_scale)
        if abs(compose_scale - 1) > 1e-1:
            img = cv.resize(src=full_img, dsize=None, fx=compose_scale, fy=compose_scale,
                            interpolation=cv.INTER_LINEAR_EXACT)
        else:
            img = full_img

        _img_size = (img.shape[1], img.shape[0])
        K = cameras[idx].K().astype(np.float32)
        R = cameras[idx].R.astype(np.float32)
        corner, image_warped = warper.warp(img, K, R, cv.INTER_LINEAR,
                                           cv.BORDER_REFLECT)
        image_warped_s = image_warped.astype(np.int16)
        mask_warped = cv.UMat(masks_warped[idx])

        if blender is None:  # no timelapse
            blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
            dst_sz = cv.detail.resultRoi(corners=corners, sizes=sizes)
            blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * blend_strength / 100
            if blend_width < 1:
                blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
            elif blend_type == "multiband":
                blender = cv.detail_MultiBandBlender()
                blender.setNumBands((np.log(blend_width) / np.log(2.) - 1.).astype(np.int32))
            elif blend_type == "feather":
                blender = cv.detail_FeatherBlender()
                blender.setSharpness(1. / blend_width)
            blender.prepare(dst_sz)
        blender.feed(cv.UMat(image_warped_s), mask_warped, corners[idx])

    result = None
    result_mask = None
    result, result_mask = blender.blend(result, result_mask)
    zoom_x = 600.0 / result.shape[1]
    dst = cv.normalize(src=result, dst=None, alpha=255., norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    dst = cv.resize(dst, dsize=None, fx=zoom_x, fy=zoom_x)

    stitch_configs = dict()
    stitch_configs['warper'] = warper
    stitch_configs['blend_strength'] = blend_strength
    stitch_configs['blend_type'] = blend_type

    return dst, stitch_configs


# more or less the bare basics required to stitch the images
def stitch_frame(all_imgs, camera_configs, stitch_configs):
    cameras = camera_configs['cameras']
    corners = camera_configs['corners']
    sizes = camera_configs['sizes']
    masks_warped = camera_configs['masks']

    warper = stitch_configs['warper']
    blend_strength = stitch_configs['blend_strength']
    blend_type = stitch_configs['blend_type']

    blender = None

    for idx in range(0, len(all_imgs)):
        img = all_imgs[idx]
        K = cameras[idx].K().astype(np.float32)
        R = cameras[idx].R.astype(np.float32)
        corner, image_warped = warper.warp(img, K, R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
        image_warped_s = image_warped.astype(np.int16)
        mask_warped = cv.UMat(masks_warped[idx])

        if blender is None:  # no timelapse
            blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
            dst_sz = cv.detail.resultRoi(corners=corners, sizes=sizes)
            blend_width = np.sqrt(dst_sz[2] * dst_sz[3]) * blend_strength / 100
            if blend_width < 1:
                blender = cv.detail.Blender_createDefault(cv.detail.Blender_NO)
            elif blend_type == "multiband":
                blender = cv.detail_MultiBandBlender()
                blender.setNumBands((np.log(blend_width) / np.log(2.) - 1.).astype(np.int32))
            elif blend_type == "feather":
                blender = cv.detail_FeatherBlender()
                blender.setSharpness(1. / blend_width)
            blender.prepare(dst_sz)
        blender.feed(cv.UMat(image_warped_s), mask_warped, corners[idx])

    result = None
    result_mask = None
    result, result_mask = blender.blend(result, result_mask)
    zoom_x = 600.0 / result.shape[1]
    dst = cv.normalize(src=result, dst=None, alpha=255., norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
    dst = cv.resize(dst, dsize=None, fx=zoom_x, fy=zoom_x)

    return dst


def do_everything(all_vids, camera_configs, output_path, fps, mtxs, dists):
    caps = []
    stitched_frames = []

    for vid_path in all_vids:
        caps.append(cv.VideoCapture(vid_path))

    frame_count = 0
    stitch_configs = None

    while caps[0].isOpened():
        print('Processing frame {}'.format(frame_count))
        ret = []
        frames = []
        for cap in caps:  # read through all videos
            ret_temp, frame_temp = cap.read()
            ret.append(ret_temp)
            frames.append(frame_temp)

        if not all(ret):
            print("Reached end of stream. Exiting ...")
            break

        frames = undistort(frames, mtxs, dists)
        # frames = undistort_fisheye(frames, mtxs, dists)  # TODO: replace line above with this if using fisheye camera model

        if stitch_configs is None:
            stitched, stitch_configs = first_stitch(frames, camera_configs)
        else:
            stitched = stitch_frame(frames, camera_configs, stitch_configs)

        stitched_frames.append(stitched)
        frame_count += 1

    for cap in caps:
        cap.release()

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'DIVX')
    vid_size = (stitched_frames[0].shape[1], stitched_frames[0].shape[0])
    out = cv.VideoWriter(output_path, fourcc, fps, vid_size)
    for frame in stitched_frames:
        out.write(frame)
    out.release()

    # return stitched_frames


def read_calibration(csv_path, num_cams):  # NOTE: also works for reading in fisheye calibration CSVs
    mtxs = []
    dists = []
    with open(csv_path, newline='') as csv_file:
        reader = csv.reader(csv_file)
        for i in range(0, num_cams):
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

    return mtxs, dists


def undistort(all_imgs, mtxs, dists):
    undistorted_imgs = []
    for i in range(0, len(all_imgs)):
        img = all_imgs[i]
        mtx = mtxs[i]
        dist = dists[i]
        h, w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        undistorted = cv.undistort(img, mtx, dist, None, newcameramtx)
        x, y, w, h = roi
        undistorted = undistorted[y:y + h, x:x + w]
        undistorted_imgs.append(undistorted)

    return undistorted_imgs


def undistort_fisheye(all_imgs, mtxs, dists):
    undistorted_imgs = []
    for i in range(0, len(all_imgs)):
        img = all_imgs[i]
        var_K = mtxs[i]
        var_D = dists[i]
        h, w = img.shape[:2]
        map1, map2 = cv.fisheye.initUndistortRectifyMap(var_K, var_D, np.eye(3), var_K, (w, h), cv.CV_16SC2)
        undistorted = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
        undistorted_imgs.append(undistorted)

    return undistorted_imgs


def main():
    # -------------File I/O Stuff-------------
    csv_path = filedialog.askopenfilename(title='Select calibration CSV:', initialfile='calibration.csv',
                                          filetypes=(('CSV', '*.csv'), ('All files', '*.*')))
    cameraConfigs_path = filedialog.askopenfilename(title='Select camera configs:', initialfile='cameraConfigs.cfg',
                                                    filetypes=(('cfg', '*.cfg'), ('All files', '*.*')))
    masks_path = filedialog.askopenfilename(title='Select masks:', initialfile='masks.yml',
                                            filetypes=(('yml', '*.yml'), ('All files', '*.*')))
    all_vids = filedialog.askopenfilenames(title='Select videos to stitch:')
    output_vid = filedialog.asksaveasfilename(title='Save stitched video:', initialfile='stitched.avi',
                                              filetypes=(('avi', '*.avi'), ('All files', '*.*')))
    # ----------------------------------------

    num_cams = len(all_vids)
    mtxs, dists = read_calibration(csv_path, num_cams)
    camera_configs = read_configs(cameraConfigs_path, masks_path)

    fps = 30  # NOTE: change video framerate here

    do_everything(all_vids, camera_configs, output_vid, fps, mtxs, dists)


if __name__ == '__main__':
    main()
