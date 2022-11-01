import cv2 as cv
import numpy as np

def stitch_frames(all_imgs): # pass in list of images
    ba = cv.detail_BundleAdjusterReproj() # --ba reproj
    ba_refine_mask = 'xxx_x' # --ba_refine_mask xxx_x
    warp_type = 'plane' # --warp plane

    # ----- default values -----
    work_megapix = 0.6
    seam_megapix = 0.1
    compose_megapix = -1
    conf_thresh = 1.0
    wave_correct = cv.detail.WAVE_CORRECT_HORIZ
    blend_type = 'multiband'
    blend_strength = 5
    finder = cv.ORB.create()
    seam_work_aspect = 1
    try_cuda = False
    matcher_type = 'homography'
    match_conf = 0.3
    range_width = -1
    expos_comp_type = cv.detail.ExposureCompensator_GAIN_BLOCKS
    expos_comp_nr_feeds = 1
    expos_comp_block_size = 32
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

