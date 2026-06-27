import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import random
import pandas as pd
from utils.augmentations import AugmentImage as apply
from utils.augmentations import AugmentBoundingBox as augmentbox
cv.setNumThreads(1)

def augment_data(image, bboxes,
                  img_size = 640,
                  p_scale = 1.0,  scale_factor = 0.9,
                  p_trans = 1.0, translate_factor = 0.1,
                  p_rot = 0.3, rotation_angle = 45.0,
                  p_shear = 0.0, shear_angle = 10.0,
                  p_hflip = 0.5,
                  p_vflip = 0.0,
                  p_mixup = 0.3,
                  p_mosaic = 0.3,
                  p_hsv = 1.0, hgain = 0.015, sgain=0.7, vgain=0.4,
                  p_grey = 0.1,
                  p_blur = 0.1,
                  p_clahe = 0.1,
                  p_cutout = 0.0,
                  p_shuffle = 0.1,
                  p_post = 0.1,
                  mode = 'test',
                  seed = None,
                  all_images = None,       # Pre-loaded images
                  all_bboxes = None,       # Pre-loaded bboxes
                  all_image_paths = None,  # Keep for backward compatibility
                  all_label_paths = None,  # Keep for backward compatibility
                  annotations_csv = None,  # Keep for backward compatibility
                  img_dir = None,
                  label_dir = None):
    """
    Apply various transformations to an input image.
    """

    if mode == 'train':
        # convert rgb to bgr
        image = apply.rgb_to_bgr(image)
        # resize image
        image = apply.resize(image, img_size)
        height, width = image.shape[:2]
        # conver to opencv format for augmentations
        bboxes = augmentbox.yolo_to_opencv(bboxes, width, height)
        # scale
        image, bboxes = apply.scale(image, bboxes, scale_factor = scale_factor, p = p_scale, seed = seed)
        # translate
        image, bboxes = apply.translate(image, bboxes, translate_factor = translate_factor, p = p_trans, seed = seed)
        # horizontal flip image
        image, bboxes = apply.horizontal_flip(image, bboxes, p = p_hflip, seed = seed)
        # vertical flip image
        # not implemented
        # image = apply.vertical_flip(image, bboxes, p = p_vflip)

        image, bboxes = apply.rotate(image, bboxes, rotation_angle = rotation_angle, p = p_rot, seed = seed)
        image, bboxes = apply.shear(image, bboxes, shear_angle = shear_angle, p = p_shear, seed = seed)
        image = apply.posterize(image, p = p_post, seed = seed)
        image = apply.greyscale(image, p = p_grey, seed = seed)
        image = apply.blur(image, p = p_blur, seed = seed)
        image = apply.clahe(image, p = p_clahe, seed = seed)
        image = apply.cutout(image, p = p_cutout, seed = seed)
        image = apply.hsv_augment(image, hgain = hgain, sgain = sgain, vgain = vgain, p = p_hsv, seed = seed)
        image = apply.shuffle_channel(image, p = p_shuffle, seed = seed)

        # mosaic and mixup overwrites existing image and augmentations
        # above augments are applied in mosaic and mixup themselves
        # apply either mosaic or mixup only if pre-loaded data is available
        if all_images is not None and all_bboxes is not None:  # ← CHECK FOR PRE-LOADED DATA
            mosaic_or_mixup = np.random.choice(["mosaic", "mixup"])
            if mosaic_or_mixup == "mosaic":
                image, bboxes = apply.mosaic(image, bboxes,
                                            p_scale, scale_factor,
                                            p_trans, translate_factor,
                                            p_rot, rotation_angle,
                                            p_shear, shear_angle,
                                            p_hflip,
                                            p_hsv, hgain, sgain, vgain,
                                            p_grey,
                                            p_blur,
                                            p_clahe,
                                            p_cutout,
                                            p_shuffle,
                                            p_post,
                                            img_size = img_size,
                                            all_images = all_images,      # ← PASS PRE-LOADED IMAGES
                                            all_bboxes = all_bboxes,     # ← PASS PRE-LOADED BBOXES
                                            img_dir = img_dir,
                                            label_dir = label_dir, 
                                            p = p_mosaic)
            else:
                image, bboxes = apply.mixup(image, bboxes,
                                            p_scale, scale_factor,
                                            p_trans, translate_factor,
                                            p_rot, rotation_angle,
                                            p_shear, shear_angle,
                                            p_hflip,
                                            p_hsv, hgain, sgain, vgain,
                                            p_grey,
                                            p_blur,
                                            p_clahe,
                                            p_cutout,
                                            p_shuffle,
                                            p_post,
                                            img_size = img_size,
                                            all_images = all_images,      # ← PASS PRE-LOADED IMAGES
                                            all_bboxes = all_bboxes,     # ← PASS PRE-LOADED BBOXES
                                            img_dir = img_dir,
                                            label_dir = label_dir, 
                                            p = p_mixup)
        # Fallback to path-based approach if pre-loaded data not available
        elif all_image_paths is not None and all_label_paths is not None:
            mosaic_or_mixup = np.random.choice(["mosaic", "mixup"])
            if mosaic_or_mixup == "mosaic":
                image, bboxes = apply.mosaic(image, bboxes,
                                            p_scale, scale_factor,
                                            p_trans, translate_factor,
                                            p_rot, rotation_angle,
                                            p_shear, shear_angle,
                                            p_hflip,
                                            p_hsv, hgain, sgain, vgain,
                                            p_grey,
                                            p_blur,
                                            p_clahe,
                                            p_cutout,
                                            p_shuffle,
                                            p_post,
                                            all_image_paths = all_image_paths,
                                            all_label_paths = all_label_paths,
                                            annotations_csv = annotations_csv,
                                            img_dir = img_dir,
                                            label_dir = label_dir, 
                                            p = p_mosaic)
            else:
                image, bboxes = apply.mixup(image, bboxes,
                                            p_scale, scale_factor,
                                            p_trans, translate_factor,
                                            p_rot, rotation_angle,
                                            p_shear, shear_angle,
                                            p_hflip,
                                            p_hsv, hgain, sgain, vgain,
                                            p_grey,
                                            p_blur,
                                            p_clahe,
                                            p_cutout,
                                            p_shuffle,
                                            p_post,
                                            all_image_paths = all_image_paths,
                                            all_label_paths = all_label_paths,
                                            annotations_csv = annotations_csv,
                                            img_dir = img_dir,
                                            label_dir = label_dir, 
                                            p = p_mixup)

    elif mode == 'test':
        # convert rgb to bgr
        image = apply.rgb_to_bgr(image)
        # resize image
        image = apply.resize(image, img_size)
        height, width = image.shape[:2]
        # conver to opencv format for augmentations
        bboxes = augmentbox.yolo_to_opencv(bboxes, width, height)

    image = apply.resize(image, width)
    # convert brg to rgb
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    # normalise # consistent with ImageNet backbone
    image = apply.normalize(image)
    # convert opencv pascalvoc format to yolo
    bboxes = augmentbox.opencv_to_yolo(bboxes, width, height)

    return image, bboxes