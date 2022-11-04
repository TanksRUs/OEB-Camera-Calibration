import cv2 as cv
import numpy as np
from collections import OrderedDict

def generate_configs(all_imgs): # pass in list of images
    ba = cv.detail_BundleAdjusterReproj() # --ba reproj
    ba_refine_mask = 'xxx_x' # --ba_refine_mask xxx_x
    warp_type = 'plane' # --warp plane

    # ----- default values -----
    blend_strength = 5
    blend_type = 'multiband'
    compose_megapix = -1
    conf_thresh = 1.0
    estimator = 'homography'
    expos_comp = 'gain_blocks'
    expos_comp_block_size = 32
    expos_comp_nr_feeds = 1
    expos_comp_type = cv.detail.ExposureCompensator_GAIN_BLOCKS
    features = 'orb'
    finder = cv.ORB.create()
    match_conf = None
    matcher_type = 'homography'
    range_width = -1
    seam = 'dp_color'
    seam_megapix = 0.1
    seam_work_aspect = 1
    try_cuda = False
    wave_correct = cv.detail.WAVE_CORRECT_HORIZ
    work_megapix = 0.6
    # --------------------------

    full_img_sizes = []
    features = []
    images = []
    is_work_scale_set = False
    is_seam_scale_set = False
    is_compose_scale_set = False
    for full_img in all_imgs:
        full_img_sizes.append((full_img.shape[1], full_img.shape[0]))
        if is_work_scale_set is False:
            work_scale = min(1.0, np.sqrt(work_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
            is_work_scale_set = True
        img = cv.resize(src=full_img, dsize=None, fx=work_scale, fy=work_scale, interpolation=cv.INTER_LINEAR_EXACT)
        if is_seam_scale_set is False:
            seam_scale = min(1.0, np.sqrt(seam_megapix * 1e6 / (full_img.shape[0] * full_img.shape[1])))
            seam_work_aspect = seam_scale / work_scale
            is_seam_scale_set = True
        img_feat = cv.detail.computeImageFeatures2(finder, img)
        features.append(img_feat)
        img = cv.resize(src=full_img, dsize=None, fx=seam_scale, fy=seam_scale, interpolation=cv.INTER_LINEAR_EXACT)
        images.append(img)

    matcher = cv.detail_BestOf2NearestMatcher(try_cuda, match_conf)
    p = matcher.apply2(features)
    matcher.collectGarbage()

    indices = cv.detail.leaveBiggestComponent(features, p, conf_thresh)
    img_subset = []
    full_img_sizes_subset = []
    for i in range(len(indices)):
        img_subset.append(images[indices[i]])
        full_img_sizes_subset.append(full_img_sizes[indices[i]])

    images = img_subset
    full_img_sizes = full_img_sizes_subset
    num_images = len(images)
    estimator = cv.detail_HomographyBasedEstimator()
    b, cameras = estimator.apply(features, p, None)

    for cam in cameras:
        cam.R = cam.R.astype(np.float32)

    adjuster = ba
    adjuster.setConfThresh(conf_thresh)

    refine_mask = np.zeros((3, 3), np.uint8)
    if ba_refine_mask[0] == 'x':
        refine_mask[0, 0] = 1
    if ba_refine_mask[1] == 'x':
        refine_mask[0, 1] = 1
    if ba_refine_mask[2] == 'x':
        refine_mask[0, 2] = 1
    if ba_refine_mask[3] == 'x':
        refine_mask[1, 1] = 1
    if ba_refine_mask[4] == 'x':
        refine_mask[1, 2] = 1
    adjuster.setRefinementMask(refine_mask)
    b, cameras = adjuster.apply(features, p, cameras)

    focals = []
    for cam in cameras:
        focals.append(cam.focal)
    focals.sort()
    if len(focals) % 2 == 1:
        warped_image_scale = focals[len(focals) // 2]
    else:
        warped_image_scale = (focals[len(focals) // 2] + focals[len(focals) // 2 - 1]) / 2

    rmats = [] # only if wave_correct is not None
    for cam in cameras:
        rmats.append(np.copy(cam.R))
    rmats = cv.detail.waveCorrect(rmats, wave_correct)
    for idx, cam in enumerate(cameras):
        cam.R = rmats[idx]

    corners = []
    masks_warped = []
    images_warped = []
    sizes = []
    masks = []
    for i in range(0, num_images):
        um = cv.UMat(255 * np.ones((images[i].shape[0], images[i].shape[1]), np.uint8))
        masks.append(um)

    warper = cv.PyRotationWarper(warp_type, warped_image_scale * seam_work_aspect)  # warper could be nullptr?
    for idx in range(0, num_images):
        K = cameras[idx].K().astype(np.float32)
        swa = seam_work_aspect
        K[0, 0] *= swa
        K[0, 2] *= swa
        K[1, 1] *= swa
        K[1, 2] *= swa
        corner, image_wp = warper.warp(images[idx], K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
        corners.append(corner)
        sizes.append((image_wp.shape[1], image_wp.shape[0]))
        images_warped.append(image_wp)
        p, mask_wp = warper.warp(masks[idx], K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
        masks_warped.append(mask_wp.get())

    images_warped_f = []
    for img in images_warped:
        imgf = img.astype(np.float32)
        images_warped_f.append(imgf)

    compensator = cv.detail.ExposureCompensator_createDefault(expos_comp_type)
    compensator.feed(corners=corners, images=images_warped, masks=masks_warped)

    seam_finder = cv.detail_DpSeamFinder('COLOR')
    masks_warped = seam_finder.find(images_warped_f, corners, masks_warped)
    compose_scale = 1
    corners = []
    sizes = []
    blender = None

    for idx in range(0, len(images)):
        full_img = all_imgs[idx]
        if not is_compose_scale_set:
            is_compose_scale_set = True
            compose_work_aspect = compose_scale / work_scale
            warped_image_scale *= compose_work_aspect
            warper = cv.PyRotationWarper(warp_type, warped_image_scale)
            for i in range(0, len(images)):
                cameras[i].focal *= compose_work_aspect
                cameras[i].ppx *= compose_work_aspect
                cameras[i].ppy *= compose_work_aspect
                sz = (int(round(full_img_sizes[i][0] * compose_scale)),
                      int(round(full_img_sizes[i][1] * compose_scale)))
                K = cameras[i].K().astype(np.float32)
                roi = warper.warpRoi(sz, K, cameras[i].R)
                corners.append(roi[0:2])
                sizes.append(roi[2:4])

        if abs(compose_scale - 1) > 1e-1:
            img = cv.resize(src=full_img, dsize=None, fx=compose_scale, fy=compose_scale,
                            interpolation=cv.INTER_LINEAR_EXACT)
        else:
            img = full_img

        _img_size = (img.shape[1], img.shape[0])
        K = cameras[idx].K().astype(np.float32)
        corner, image_warped = warper.warp(img, K, cameras[idx].R, cv.INTER_LINEAR, cv.BORDER_REFLECT)
        mask = 255 * np.ones((img.shape[0], img.shape[1]), np.uint8)
        p, mask_warped = warper.warp(mask, K, cameras[idx].R, cv.INTER_NEAREST, cv.BORDER_CONSTANT)
        compensator.apply(idx, corners[idx], image_warped, mask_warped)
        image_warped_s = image_warped.astype(np.int16)
        dilated_mask = cv.dilate(masks_warped[idx], None)
        seam_mask = cv.resize(dilated_mask, (mask_warped.shape[1], mask_warped.shape[0]), 0, 0, cv.INTER_LINEAR_EXACT)
        mask_warped = cv.bitwise_and(seam_mask, mask_warped)  # TODO: figure out if this is the mask I should be saving

        if blender is None: # no timelapse
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
    cv.imshow('Stitched Image', dst)
    cv.waitKey()
    cv.destroyAllWindows()

    ret_val = OrderedDict()
    ret_val['cameras'] = cameras
    ret_val['masks_warped'] = masks_warped

    return ret_val

def main():
    # im1 = cv.imread("C:/Users/duanr/Desktop/Video Stitching/Oct 23 Configuration/1.png")
    # im2 = cv.imread("C:/Users/duanr/Desktop/Video Stitching/Oct 23 Configuration/2.png")
    # im3 = cv.imread("C:/Users/duanr/Desktop/Video Stitching/Oct 23 Configuration/3.png")
    # im4 = cv.imread("C:/Users/duanr/Desktop/Video Stitching/Oct 23 Configuration/4.png")
    im1 = cv.imread("C:\\Users\\richa\\OneDrive - University of Waterloo\\Desktop\\NRC\\Python\\OEB-Camera-Calibration\\Oct 21 Test\\Original Images\\1.png")
    im2 = cv.imread("C:\\Users\\richa\\OneDrive - University of Waterloo\\Desktop\\NRC\\Python\\OEB-Camera-Calibration\\Oct 21 Test\\Original Images\\2.png")
    im3 = cv.imread("C:\\Users\\richa\\OneDrive - University of Waterloo\\Desktop\\NRC\\Python\\OEB-Camera-Calibration\\Oct 21 Test\\Original Images\\3.png")
    im4 = cv.imread("C:\\Users\\richa\\OneDrive - University of Waterloo\\Desktop\\NRC\\Python\\OEB-Camera-Calibration\\Oct 21 Test\\Original Images\\4.png")

    all_imgs = [im1, im2, im3, im4]
    generate_configs(all_imgs)

if __name__ == '__main__':
    main()