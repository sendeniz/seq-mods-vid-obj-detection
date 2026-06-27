import cv2 as cv
import numpy as np
import pandas as pd
import random
import os
from PIL import Image

cv.setNumThreads(1)

class AugmentImage:
    @staticmethod
    def normalize(image, mean = [0, 0 , 0], std = [1, 1, 1], max_pixel_val = 255):
        image = image.astype(np.float32)
        image = (image - np.array(mean, dtype=np.float32) * max_pixel_val) / (np.array(std, dtype=np.float32) * max_pixel_val)
        return image

    def posterize(image,  num_bits = 4, p = 0.1, seed = None):
        if seed is not None:
            np.random.seed(seed)

        if np.random.rand() <= p:
            levels = 2 ** num_bits
            divisor = 256 // levels
            lut = np.arange(256, dtype=np.uint8) // divisor * divisor
            image = cv.LUT(image, lut)
        return image

    def greyscale(image, p = 0.1, seed = None):
        if seed is not None:
            np.random.seed(seed)

        if np.random.rand() <= p:
            image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            image = cv.merge([image, image, image])
        return image

    def blur(image, p = 0.1, seed = None):
        if seed is not None:
            np.random.seed(seed)

        if np.random.rand() <= p:
            kernel_width = np.random.randint(3, 7)
            kernel_height = np.random.randint(3, 7)
            kernel_width = kernel_width + 1 if kernel_width % 2 == 0 else kernel_width
            kernel_height = kernel_height + 1 if kernel_height % 2 == 0 else kernel_height
            std_deviation = np.random.uniform(0.2, 1.0)
            image = cv.GaussianBlur(image, (kernel_width, kernel_height), std_deviation)
        return image

    def clahe(image, p = 0.1, seed = None):
        if seed is not None:
            np.random.seed(seed)

        if np.random.rand() <= p:
            b, g, r = cv.split(image)
            # apply clahe to each color channel independently
            clahe = cv.createCLAHE(clipLimit = 4.0, tileGridSize=(8, 8))
            b_clahe = clahe.apply(b)
            g_clahe = clahe.apply(g)
            r_clahe = clahe.apply(r)
            # merge the clahe-enhanced channels back together
            image = cv.merge([b_clahe, g_clahe, r_clahe])
        return image

    def cutout(image, p = 0.3, seed = None):
        if seed is not None:
            np.random.seed(seed)

        if np.random.rand() <= p:
            height, width = image.shape[:2]
            scales = [0.10] * 1 + [0.080] * 2 + [0.060] * 4 + [0.040] * 8 + [0.020] * 16  # image size fraction
            for s in scales:
                mask_h = np.random.randint(1, int(height * s))  # create random masks
                mask_w = np.random.randint(1, int(width * s))
                # box
                xmin = max(0, np.random.randint(0, width) - mask_w // 2)
                ymin = max(0, np.random.randint(0, height) - mask_h // 2)
                xmax = min(width, xmin + mask_w)
                ymax = min(height, ymin + mask_h)
                # apply random color mask
                image[ymin:ymax, xmin:xmax] = [np.random.randint(64, 191) for _ in range(3)]
        return image

    def hsv_augment(image, hgain = 0.1, sgain=0.9, vgain=0.9, p = 1.0, seed = None):
        if seed is not None:
            np.random.seed(seed)

        if np.random.rand() <= p:
            r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
            hue, sat, val = cv.split(cv.cvtColor(image, cv.COLOR_BGR2HSV))
            dtype = image.dtype  # uint8

            x = np.arange(0, 256, dtype=np.int16)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            image_hsv = cv.merge((cv.LUT(hue, lut_hue), cv.LUT(sat, lut_sat), cv.LUT(val, lut_val))).astype(dtype)
            image = cv.cvtColor(image_hsv, cv.COLOR_HSV2BGR, dst=image)
        return image

    def scale(image, bboxes, scale_factor = 1.09, lower_bound = 1.0, p = 1.0, preserve_ratio = True, seed = None):
        """
        Scales an image by a random factor bound below by the lower_bound. 

        Args:
        - image (numpy.ndarray):
            The input image to be scaled.
        - scale_factor (float):
            The maximum ratio to which the image is scaled. Default is 1.09.
        - lower_bound (float):
            The minimum ratio to which the image is scaled. Default is original image size: 1.00.

        Returns:
        - numpy.ndarray:
            The scaled image.
        """
        if seed is not None:
            np.random.seed(seed)

        if np.random.rand() <= p:
            height, width =  image.shape[:2]
            if preserve_ratio == True:
                scale_x = np.random.uniform(lower_bound, scale_factor)
                scale_y = scale_x
            elif preserve_ratio == False:
                scale_x = np.random.uniform(lower_bound, scale_factor)
                scale_x = np.random.uniform(lower_bound, scale_factor)

            image = cv.resize(image, None, fx = scale_x, fy = scale_y)
            # compute scale augmented bounding boxes
            bboxes[:,:4] *= [scale_x, scale_y, scale_x, scale_y]
            # clip bounding boxes to be within image dimension
            # and drop bounding boxes based on min visbility of 0.4 compared to orginal
            bboxes = AugmentBoundingBox.clip(bboxes, img_size = height, min_visibility = 0.4)
            # create black image for padding if scaled below 1.0
            blankimg = np.zeros((height, width, 3), dtype=np.uint8)
            y_lim = int(min(scale_y, 1) * height)
            x_lim = int(min(scale_x, 1) * width)
            # paste scale image on black background
            # if scaled down padded background will be visible
            blankimg[:y_lim,:x_lim,:] = image[:y_lim,:x_lim,:]
            image = blankimg
        return image, bboxes

    def translate(image, bboxes, translate_factor = 0.09, p = 1.0, preserve_ratio = False, seed = None):
        """
        Translates an image by a random factor representing the percentage of the image dimensions.

        Args:
        - image (numpy.ndarray):
            The input image to be scaled.
        - scale_factor (float):
            The maximum percentage to which the image is translated. Default is 9%: 0.09.
            The minimum percentage to which the image is scaled is the negative scale_factor.
        Returns:
        - numpy.ndarray:
            The translated image.
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        if np.random.rand() <= p:
            lower_bound = - translate_factor
            height, width =  image.shape[:2]
            if preserve_ratio == False:
                translate_x_val = random.uniform(lower_bound, translate_factor)
                translate_y_val = random.uniform(lower_bound, translate_factor)
            elif preserve_ratio == True:
                translate_x_val = random.uniform(lower_bound, translate_factor)
                translate_y_val = translate_x_val
            # create blank image for padding
            blankimg = np.zeros((height, width, 3), dtype=np.uint8)
            corner_x = int(translate_x_val * width)
            corner_y = int(translate_y_val * height)
            # change origin to the top-left corner of the translated box
            orig_box_cords =  [max(0,corner_y), max(corner_x,0), min(height, corner_y + height), min(width, corner_x + width)]
            mask = image[max(-corner_y, 0):min(height, -corner_y + height), max(-corner_x, 0):min(width, -corner_x + width),:]
            blankimg[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3],:] = mask
            # translate bounding boxes
            bboxes[:,:4] += [corner_x, corner_y, corner_x, corner_y]
            # clip bounding boxes to be within image dimension
            # and drop bounding boxes based on min visbility of 0.4 compared to orginal
            bboxes = AugmentBoundingBox.clip(bboxes, img_size = height, min_visibility = 0.4)
            image = blankimg
        return image, bboxes

    def horizontal_flip(image, bboxes, p = 0.3, seed = None):
        if seed is not None:
            np.random.seed(seed)

        if np.random.rand() <= p:
            image = cv.flip(image, 1)
            height, width = image.shape[:2]
            center_width = width / 2
            # since we flip horizontally only center width is needed
            bboxes[:,0] += 2 * (center_width - bboxes[:,0])
            bboxes[:,2] += 2 * (center_width - bboxes[:,2])
            bboxes_width = abs(bboxes[:,0] - bboxes[:,2])
            bboxes[:,0] -= bboxes_width
            bboxes[:,2] += bboxes_width
        return image, bboxes

    def vertical_flip(image, bboxes, p = 0.1, seed = None):
        if seed is not None:
            np.random.seed(seed)

        if np.random.rand() <= p:
            image = cv.flip(image, 0)
        return image, bboxes

    def shear(image, bboxes, shear_angle = 10.0, p = 0.3, seed = None):
        """
        Sheares an image by a random factor represting the degrees.

        Args:
        - image (numpy.ndarray):
            The input image to be scaled.
        - shear_angle (float):
            The maximum degrees by which the image will be sheared. Default is 10 degrees.

        Returns:
        - numpy.ndarray:
            The scaled image.
        - sheared_width (float):
            The shared image width.
        - shared_height (float):
             The shared image height.
        """
        if seed is not None:
            np.random.seed(seed)

        if np.random.rand() <= p:
            shear_angle = np.random.uniform(-shear_angle / 100, shear_angle / 100)
            height, width = image.shape[:2]
            if shear_angle < 0:
                image, bboxes = AugmentImage.horizontal_flip(image, bboxes)
            # shear matrix
            M = np.array([[1, abs(shear_angle), 0], [0, 1, 0]])
            new_width = width + abs(shear_angle * height)
            bboxes[:,0] += (bboxes[:,1] * abs(shear_angle)).astype(int)
            bboxes[:,2] += (bboxes[:,3] * abs(shear_angle)).astype(int)
            image = cv.warpAffine(image, M, (int(new_width), width))
            if shear_angle < 0:
                image, bboxes = AugmentImage.horizontal_flip(image, bboxes)

            # rotation changes height width of image
            # adjust to original square size
            image = AugmentImage.resize(image, width)
            scale_x = new_width / width
            bboxes[:,:4] /=  [scale_x, 1, scale_x, 1]
        return image, bboxes

    def rotate(image, bboxes, rotation_angle = 45, p = 0.3, seed = None):
        if seed is not None:
            np.random.seed(seed)

        if np.random.rand() <= p:
            height, width = image.shape[:2]
            center_x, center_y = width // 2, height // 2
            rotation_angle = np.random.uniform(-rotation_angle, rotation_angle)
            # define rotation matrix
            M = cv.getRotationMatrix2D((center_x, center_y), rotation_angle, 1.0)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            # new width and height of rotated image
            rotated_width = int((height * sin) + (width * cos))
            rotated_height = int((height * cos) + (width * sin))
            # adjust to account for translation when rotated
            M[0, 2] += (rotated_width / 2) - center_x
            M[1, 2] += (rotated_height / 2) - center_y
            # rotate image
            image = cv.warpAffine(image, M, (rotated_width, rotated_height))
            new_height, new_width = image.shape[:2]
            corners = AugmentBoundingBox.compute_corners(bboxes)
            # extract class c
            c = corners[:,8]
            corners = corners[:,:8].reshape(-1,2)

            ones = np.ones((corners.shape[0], 1))
            corners = np.hstack((corners, ones))
            # rotate corners
            corners = np.dot(M, corners.T).T
            corners = corners.reshape(-1,8)
            corners = np.hstack((corners, c[:, None]), dtype=np.float32)
            # convert corners to opencv format/pascal voc format
            bboxes = AugmentBoundingBox.corner_to_opencv(corners)
            scale_x = new_width / width
            scale_y = new_height / height
            image  = AugmentImage.resize(image, width)
            bboxes[:,:4] /= [scale_x, scale_y, scale_x, scale_y]
            bboxes = AugmentBoundingBox.clip(bboxes, img_size = height, min_visibility = 0.4)
        return image, bboxes

    def shuffle_channel(image , p = 0.1, seed = None):
        if seed is not None:
            np.random.seed(seed)

        if np.random.rand() <= p:
            b, g, r = cv.split(image)
            # shuffle the channels in a random order
            channel_order = np.random.permutation([0, 1, 2])
            # reorder the channels based on the random order
            image = cv.merge([b, g, r])[..., channel_order]
        return image

    def resize(image, resize_to):
        image = cv.resize(image, (resize_to, resize_to))
        return image

    def rgb_to_bgr(image):
        image = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
        return image

    def mosaic(image, bboxes,
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
       img_size=640,
       all_images=None,        # NEW: Pre-loaded images
       all_bboxes=None,       # NEW: Pre-loaded bboxes
       all_image_paths=None,
       all_label_paths=None,
       annotations_csv=None,
       img_dir=None, 
       label_dir=None, 
       p=0.3):

        if np.random.rand() <= p:
            mosaic_images = []
            bboxes_lst = []
            bboxes_len = []
            
            mosaic_size = np.random.choice([4, 9])
            
            # USE PRE-LOADED DATA IF AVAILABLE (FAST)
            if all_images is not None and all_bboxes is not None:
                mosaic_indices = random.sample(range(len(all_images)), mosaic_size)

                for idx in mosaic_indices:
                    # Use pre-loaded data directly - NO DISK I/O!
                    image_mosaic = all_images[idx].copy()  # Copy to avoid modifying original
                    bboxes_mosaic = all_bboxes[idx].copy()
                    
                    # Convert to opencv format
                    resize_to = img_size // (3 if mosaic_size == 9 else 2)
                    image_mosaic = AugmentImage.resize(image_mosaic, resize_to)
                    height, width = image_mosaic.shape[:2]
                    bboxes_mosaic = AugmentBoundingBox.yolo_to_opencv(bboxes_mosaic, width, height)

                    # Apply augmentations
                    image_mosaic, bboxes_mosaic = AugmentImage.scale(image_mosaic, bboxes_mosaic, scale_factor, p=p_scale)
                    image_mosaic, bboxes_mosaic = AugmentImage.translate(image_mosaic, bboxes_mosaic, translate_factor, p=p_trans)
                    image_mosaic, bboxes_mosaic = AugmentImage.rotate(image_mosaic, bboxes_mosaic, rotation_angle, p=p_rot)
                    image_mosaic, bboxes_mosaic = AugmentImage.shear(image_mosaic, bboxes_mosaic, shear_angle, p=p_shear)
                    image_mosaic, bboxes_mosaic = AugmentImage.horizontal_flip(image_mosaic, bboxes_mosaic, p=p_hflip)
                    
                    bboxes_len.append(len(bboxes_mosaic))
                    image_mosaic = AugmentImage.hsv_augment(image_mosaic, hgain, sgain, vgain, p=p_hsv)
                    image_mosaic = AugmentImage.greyscale(image_mosaic, p=p_grey)
                    image_mosaic = AugmentImage.blur(image_mosaic, p=p_blur)
                    image_mosaic = AugmentImage.clahe(image_mosaic, p=p_clahe)
                    image_mosaic = AugmentImage.cutout(image_mosaic, p=p_cutout)
                    image_mosaic = AugmentImage.shuffle_channel(image_mosaic, p=p_shuffle)
                    image_mosaic = AugmentImage.posterize(image_mosaic, p=p_post)

                    bboxes_lst.extend(bboxes_mosaic)
                    mosaic_images.append(image_mosaic)
                    
            # FALLBACK TO DISK READING (SLOW)
            else:
                if all_image_paths is None or all_label_paths is None:
                    return image, bboxes
                    
                mosaic_indices = random.sample(range(len(all_image_paths)), mosaic_size)

                for idx in mosaic_indices:
                    label_path = os.path.join(label_dir, all_label_paths[idx])
                    img_path = os.path.join(img_dir, all_image_paths[idx])
                    bboxes_mosaic = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).astype(np.float32)
                    image_mosaic = np.array(Image.open(img_path).convert("RGB"))
                    image_mosaic = cv.cvtColor(np.array(image_mosaic), cv.COLOR_RGB2BGR)

                    # Convert to opencv format
                    resize_to = img_size // (3 if mosaic_size == 9 else 2)
                    image_mosaic = AugmentImage.resize(image_mosaic, resize_to)
                    height, width = image_mosaic.shape[:2]
                    bboxes_mosaic = AugmentBoundingBox.yolo_to_opencv(bboxes_mosaic, width, height)

                    # Apply augmentations
                    image_mosaic, bboxes_mosaic = AugmentImage.scale(image_mosaic, bboxes_mosaic, scale_factor, p=p_scale)
                    image_mosaic, bboxes_mosaic = AugmentImage.translate(image_mosaic, bboxes_mosaic, translate_factor, p=p_trans)
                    image_mosaic, bboxes_mosaic = AugmentImage.rotate(image_mosaic, bboxes_mosaic, rotation_angle, p=p_rot)
                    image_mosaic, bboxes_mosaic = AugmentImage.shear(image_mosaic, bboxes_mosaic, shear_angle, p=p_shear)
                    image_mosaic, bboxes_mosaic = AugmentImage.horizontal_flip(image_mosaic, bboxes_mosaic, p=p_hflip)
                    
                    bboxes_len.append(len(bboxes_mosaic))
                    image_mosaic = AugmentImage.hsv_augment(image_mosaic, hgain, sgain, vgain, p=p_hsv)
                    image_mosaic = AugmentImage.greyscale(image_mosaic, p=p_grey)
                    image_mosaic = AugmentImage.blur(image_mosaic, p=p_blur)
                    image_mosaic = AugmentImage.clahe(image_mosaic, p=p_clahe)
                    image_mosaic = AugmentImage.cutout(image_mosaic, p=p_cutout)
                    image_mosaic = AugmentImage.shuffle_channel(image_mosaic, p=p_shuffle)
                    image_mosaic = AugmentImage.posterize(image_mosaic, p=p_post)

                    bboxes_lst.extend(bboxes_mosaic)
                    mosaic_images.append(image_mosaic)

            # Mosaic composition (same for both paths)
            if mosaic_size == 4:
                image = np.vstack([np.hstack([mosaic_images[0], mosaic_images[1]]), 
                                np.hstack([mosaic_images[2], mosaic_images[3]])])
                bboxes_len = np.cumsum(bboxes_len).tolist()
                bboxes = np.array(bboxes_lst)

                # Bounding box adjustments
                bboxes[bboxes_len[0]:bboxes_len[1], :4] += [height, 0, height, 0]
                bboxes[bboxes_len[1]:bboxes_len[2], :4] += [0, height, 0, height]
                bboxes[bboxes_len[2]:bboxes_len[3], :4] += [height, height, height, height]

            elif mosaic_size == 9:
                image = np.vstack([np.hstack([mosaic_images[0], mosaic_images[1], mosaic_images[2]]),
                                np.hstack([mosaic_images[3], mosaic_images[4], mosaic_images[5]]),
                                np.hstack([mosaic_images[6], mosaic_images[7], mosaic_images[8]])])
                bboxes_len = np.cumsum(bboxes_len).tolist()
                bboxes = np.array(bboxes_lst)
                
                # Bounding box adjustments for 9-grid
                bboxes[bboxes_len[0]:bboxes_len[1], :4] += [height, 0, height, 0]
                bboxes[bboxes_len[1]:bboxes_len[2], :4] += [height*2, 0, height*2, 0]
                bboxes[bboxes_len[2]:bboxes_len[3], :4] += [0, height, 0, height]
                bboxes[bboxes_len[3]:bboxes_len[4], :4] += [height, height, height, height]
                bboxes[bboxes_len[4]:bboxes_len[5], :4] += [height*2, height, height*2, height]
                bboxes[bboxes_len[5]:bboxes_len[6], :4] += [0, height*2, 0, height*2]
                bboxes[bboxes_len[6]:bboxes_len[7], :4] += [height, height*2, height, height*2]
                bboxes[bboxes_len[7]:bboxes_len[8], :4] += [height*2, height*2, height*2, height*2]
                
        return image, bboxes

    def mixup(image, bboxes,
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
      img_size=640,
      all_images=None,        # NEW: Pre-loaded images
      all_bboxes=None,       # NEW: Pre-loaded bboxes
      all_image_paths=None,
      all_label_paths=None,
      annotations_csv=None,
      img_dir=None, 
      label_dir=None, 
      p=0.3):

        if np.random.rand() <= p:
            mixup_images = []
            mixup_bboxes_list = []
            
            # USE PRE-LOADED DATA IF AVAILABLE (FAST)
            if all_images is not None and all_bboxes is not None:
                mixup_indices = random.sample(range(len(all_images)), 2)

                for idx in mixup_indices:
                    # Use pre-loaded data directly - NO DISK I/O!
                    image_mixup = all_images[idx].copy()  # Copy to avoid modifying original
                    bboxes_mixup = all_bboxes[idx].copy()
                    
                    # Convert to opencv format
                    image_mixup = AugmentImage.resize(image_mixup, img_size)
                    height, width = image_mixup.shape[:2]
                    bboxes_mixup = AugmentBoundingBox.yolo_to_opencv(bboxes_mixup, width, height)

                    # Apply augmentations
                    image_mixup, bboxes_mixup = AugmentImage.scale(image_mixup, bboxes_mixup, scale_factor, p=p_scale)
                    image_mixup, bboxes_mixup = AugmentImage.translate(image_mixup, bboxes_mixup, translate_factor, p=p_trans)
                    image_mixup, bboxes_mixup = AugmentImage.rotate(image_mixup, bboxes_mixup, rotation_angle, p=p_rot)
                    image_mixup, bboxes_mixup = AugmentImage.shear(image_mixup, bboxes_mixup, shear_angle, p=p_shear)
                    image_mixup, bboxes_mixup = AugmentImage.horizontal_flip(image_mixup, bboxes_mixup, p=p_hflip)
                    
                    image_mixup = AugmentImage.hsv_augment(image_mixup, hgain, sgain, vgain, p=p_hsv)
                    image_mixup = AugmentImage.greyscale(image_mixup, p=p_grey)
                    image_mixup = AugmentImage.blur(image_mixup, p=p_blur)
                    image_mixup = AugmentImage.clahe(image_mixup, p=p_clahe)
                    image_mixup = AugmentImage.cutout(image_mixup, p=p_cutout)
                    image_mixup = AugmentImage.shuffle_channel(image_mixup, p=p_shuffle)
                    image_mixup = AugmentImage.posterize(image_mixup, p=p_post)

                    mixup_images.append(image_mixup)
                    mixup_bboxes_list.extend(bboxes_mixup)
                    
            # FALLBACK TO DISK READING (SLOW)
            else:
                if all_image_paths is None or all_label_paths is None:
                    return image, bboxes
                    
                mixup_indices = random.sample(range(len(all_image_paths)), 2)

                for idx in mixup_indices:
                    label_path = os.path.join(label_dir, all_label_paths[idx])
                    img_path = os.path.join(img_dir, all_image_paths[idx])
                    bboxes_mixup = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).astype(np.float32)
                    image_mixup = np.array(Image.open(img_path).convert("RGB"))
                    image_mixup = cv.cvtColor(np.array(image_mixup), cv.COLOR_RGB2BGR)
                    
                    # Convert to opencv format
                    image_mixup = AugmentImage.resize(image_mixup, img_size)
                    height, width = image_mixup.shape[:2]
                    bboxes_mixup = AugmentBoundingBox.yolo_to_opencv(bboxes_mixup, width, height)

                    # Apply augmentations
                    image_mixup, bboxes_mixup = AugmentImage.scale(image_mixup, bboxes_mixup, scale_factor, p=p_scale)
                    image_mixup, bboxes_mixup = AugmentImage.translate(image_mixup, bboxes_mixup, translate_factor, p=p_trans)
                    image_mixup, bboxes_mixup = AugmentImage.rotate(image_mixup, bboxes_mixup, rotation_angle, p=p_rot)
                    image_mixup, bboxes_mixup = AugmentImage.shear(image_mixup, bboxes_mixup, shear_angle, p=p_shear)
                    image_mixup, bboxes_mixup = AugmentImage.horizontal_flip(image_mixup, bboxes_mixup, p=p_hflip)
                    
                    image_mixup = AugmentImage.hsv_augment(image_mixup, hgain, sgain, vgain, p=p_hsv)
                    image_mixup = AugmentImage.greyscale(image_mixup, p=p_grey)
                    image_mixup = AugmentImage.blur(image_mixup, p=p_blur)
                    image_mixup = AugmentImage.clahe(image_mixup, p=p_clahe)
                    image_mixup = AugmentImage.cutout(image_mixup, p=p_cutout)
                    image_mixup = AugmentImage.shuffle_channel(image_mixup, p=p_shuffle)
                    image_mixup = AugmentImage.posterize(image_mixup, p=p_post)

                    mixup_images.append(image_mixup)
                    mixup_bboxes_list.extend(bboxes_mixup)
                
            # Mixup blending (same for both paths)
            lambda_mix = np.random.beta(8.0, 8.0)
            image = (mixup_images[0] * lambda_mix + mixup_images[1] * (1 - lambda_mix)).astype(np.uint8)
            bboxes = np.array(mixup_bboxes_list)
            
        return image, bboxes

    def mosaic_seq(img_size, img_dir,
                label_dir,
                vid_id_csv, 
                frame_ids_dir,
                p_scale = 1.0, scale_factor = 0.9,
                p_trans = 1.0, translate_factor = 0.1,
                p_rot = 0.3, rotation_angle = 45.0,
                p_shear = 0.0, shear_angle = 10.0,
                p_hflip = 0.5,
                p_hsv = 1.0, hgain = 0.015, sgain = 0.7, vgain = 0.4,
                p_grey = 0.1,
                p_blur = 0.1,
                p_clahe = 0.1,
                p_cutout = 0.0,
                p_shuffle = 0.1,
                p_post = 0.1,
                seq_len = 2):
    
        all_vid_frame_ids = []
        all_vid_frame_ids_paths = []
        all_sample_rates = []
        all_vid_folder_ids = []

        mosaic_size = np.random.choice([4, 9])

        for i in range(mosaic_size):
            random_index = random.randint(0, len(vid_id_csv) - 1)
            vid_folder_id = vid_id_csv.iloc[random_index, 0]
            vid_frame_ids_path = frame_ids_dir + vid_folder_id
            vid_frame_ids = pd.read_csv(f'{vid_frame_ids_path}.csv')
            sampling_rate = sampling_rate = max(1, len(vid_frame_ids) // seq_len)
            all_vid_frame_ids.append(vid_frame_ids)
            all_vid_frame_ids_paths.append(vid_frame_ids_path)
            all_sample_rates.append(sampling_rate)
            all_vid_folder_ids.append(vid_folder_id)

        final_images = []
        final_bboxes = []
        if mosaic_size == 4:    
            img_lst1, img_lst2, img_lst3, img_lst4 = [], [], [], []
            bboxes_lst1, bboxes_lst2, bboxes_lst3, bboxes_lst4 = [], [], [], []

            for j in range(0, len(all_vid_frame_ids)):
                # random seed to ensure consistent augmentation across sequence
                random_seed = np.random.randint(0, 10000)
                for i in range(0, len(all_vid_frame_ids[j]), all_sample_rates[j]):
                    label_path = os.path.join(label_dir, all_vid_folder_ids[j], all_vid_frame_ids[j].iloc[i, 0]).replace("\\", "/")
                    img_path = os.path.join(img_dir, all_vid_folder_ids[j], all_vid_frame_ids[j].iloc[i, 0]).replace("\\", "/").replace(".txt", ".JPEG")
                    bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).astype(np.float32)
                    
                    image = np.array(Image.open(img_path).convert("RGB"))
                    image = AugmentImage.rgb_to_bgr(image)
                    # resize image
                    image = AugmentImage.resize(image, img_size // 2)
                    height, width = image.shape[:2]
                    # conver to opencv format for augmentations
                    bboxes = AugmentBoundingBox.yolo_to_opencv(bboxes, width, height)
                    # scale
                    image, bboxes = AugmentImage.scale(image, bboxes, scale_factor = scale_factor, p = p_scale, seed = random_seed)
                    # translate
                    image, bboxes = AugmentImage.translate(image, bboxes, translate_factor = translate_factor, p = p_trans, seed = random_seed)
                    # horizontal flip image
                    image, bboxes = AugmentImage.horizontal_flip(image, bboxes, p = p_hflip, seed = random_seed)
                    image, bboxes = AugmentImage.rotate(image, bboxes, rotation_angle = rotation_angle, p = p_rot, seed = random_seed)
                    image, bboxes = AugmentImage.shear(image, bboxes, shear_angle = shear_angle, p = p_shear, seed = random_seed)
                    image = AugmentImage.posterize(image, p = p_post, seed = random_seed)
                    image = AugmentImage.greyscale(image, p = p_grey, seed = random_seed)
                    image = AugmentImage.blur(image, p = p_blur, seed = random_seed)
                    image = AugmentImage.clahe(image, p = p_clahe, seed = random_seed)
                    image = AugmentImage.cutout(image, p = p_cutout, seed = random_seed)
                    image = AugmentImage.hsv_augment(image, hgain = hgain, sgain = sgain, vgain = vgain, p = p_hsv, seed = random_seed)
                    image = AugmentImage.shuffle_channel(image, p = p_shuffle, seed = random_seed)
                    
                    if j == 0:
                        img_lst1.append(np.array(image))
                        bboxes_lst1.append(bboxes)
                    if j == 1:
                        img_lst2.append(np.array(image))
                        bboxes_lst2.append(bboxes)
                    if j == 2:
                        img_lst3.append(np.array(image))
                        bboxes_lst3.append(bboxes)
                    if j == 3:
                        img_lst4.append(np.array(image))
                        bboxes_lst4.append(bboxes)

            # ensure img_lists have same length
            img_lst1 = img_lst1[:seq_len] if len(img_lst1) > seq_len else img_lst1
            img_lst2 = img_lst2[:seq_len] if len(img_lst2) > seq_len else img_lst2
            img_lst3 = img_lst3[:seq_len] if len(img_lst3) > seq_len else img_lst3
            img_lst4 = img_lst4[:seq_len] if len(img_lst4) > seq_len else img_lst4

            # ensure bboxes have same length
            bboxes_lst1 = bboxes_lst1[:seq_len] if len(bboxes_lst1) > seq_len else bboxes_lst1
            bboxes_lst2 = bboxes_lst2[:seq_len] if len(bboxes_lst2) > seq_len else bboxes_lst2
            bboxes_lst3 = bboxes_lst3[:seq_len] if len(bboxes_lst3) > seq_len else bboxes_lst3
            bboxes_lst4 = bboxes_lst4[:seq_len] if len(bboxes_lst4) > seq_len else bboxes_lst4

            # repeat from beginning if sequence is smaller than intended sequence len
            for img_lst in [img_lst1, img_lst2, img_lst3, img_lst4]:
                while len(img_lst) < seq_len:
                    imgs_to_add = seq_len - len(img_lst)
                    img_lst.extend(img_lst[:imgs_to_add])

            for bboxes_lst in [bboxes_lst1, bboxes_lst2, bboxes_lst3, bboxes_lst4]:
                while len(bboxes_lst) < seq_len:
                    bboxes_to_add = seq_len - len(bboxes_lst)
                    bboxes_lst.extend(bboxes_lst[:bboxes_to_add])

            # store mosaiced img and bboxes as list to create sequence
            for k in range(len(img_lst1)):
                mosaic_img = np.vstack([np.hstack([img_lst1[k], img_lst2[k]]), np.hstack([img_lst3[k], img_lst4[k]])])
                # 1. top left image of mosaic: no adjustment so skip
                # note: height is the height of a single mosaic tile not the height of the total image
                # 2. top right image of mosaic: (x1,y1,x2,y2) + (height, 0, height, 0)
                # 3. bottom left image of mosaic: (x1,y1,x2,y2) + (0, height, 0, height)
                # 4. bottom right image of mosaic: (x1,y1,x2,y2) + (height, height, height, height)
                # 5. we add an additional 0 at the end to not change the class 
                #print(bboxes_lst2[k])
                mosaic_bbox = np.vstack([bboxes_lst1[k], bboxes_lst2[k] + [height , 0, height, 0, 0], bboxes_lst3[k] + [0, height, 0, height, 0], bboxes_lst4[k] + [height, height, height, height, 0]])
                
                # base augmentations to ensure consistency, normalization and yolo format
                mosaic_img = AugmentImage.resize(mosaic_img, img_size)
                # convert brg to rgb
                mosaic_img = cv.cvtColor(mosaic_img, cv.COLOR_BGR2RGB)
                # normalise # consistent with ImageNet backbone
                mosaic_img = AugmentImage.normalize(mosaic_img) #, mean = [0.485, 0.456, 0.406], std = [0.229, 0.244, 0.255])
                #image = Image.fromarray(image)

                # convert opencv pascalvoc format to yolo
                mosaic_bbox = AugmentBoundingBox.opencv_to_yolo(mosaic_bbox, img_size, img_size)
                final_images.append(mosaic_img)
                final_bboxes.append(mosaic_bbox)

            return final_images, final_bboxes
        
        if mosaic_size == 9:    
            img_lst1, img_lst2, img_lst3 = [], [], []
            img_lst4, img_lst5, img_lst6 = [], [], []
            img_lst7, img_lst8, img_lst9 = [], [], []

            bboxes_lst1, bboxes_lst2, bboxes_lst3 = [], [], []
            bboxes_lst4, bboxes_lst5, bboxes_lst6 = [], [], []
            bboxes_lst7, bboxes_lst8, bboxes_lst9 = [], [], []

            for j in range(0, len(all_vid_frame_ids)):
                # random seed to ensure consistent augmentation across sequence
                random_seed = np.random.randint(0, 10000)
                for i in range(0, len(all_vid_frame_ids[j]), all_sample_rates[j]):
                    label_path = os.path.join(label_dir, all_vid_folder_ids[j], all_vid_frame_ids[j].iloc[i, 0]).replace("\\", "/")
                    img_path = os.path.join(img_dir, all_vid_folder_ids[j], all_vid_frame_ids[j].iloc[i, 0]).replace("\\", "/").replace(".txt", ".JPEG")
                    bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).astype(np.float32)
                    
                    image = np.array(Image.open(img_path).convert("RGB"))
                    image = AugmentImage.rgb_to_bgr(image)
                    # resize image
                    image = AugmentImage.resize(image, img_size // 3)
                    height, width = image.shape[:2]
                    # conver to opencv format for augmentations
                    bboxes = AugmentBoundingBox.yolo_to_opencv(bboxes, width, height)
                    # scale
                    image, bboxes = AugmentImage.scale(image, bboxes, scale_factor = scale_factor, p = p_scale, seed = random_seed)
                    # translate
                    image, bboxes = AugmentImage.translate(image, bboxes, translate_factor = translate_factor, p = p_trans, seed = random_seed)
                    # horizontal flip image
                    image, bboxes = AugmentImage.horizontal_flip(image, bboxes, p = p_hflip, seed = random_seed)
                    image, bboxes = AugmentImage.rotate(image, bboxes, rotation_angle = rotation_angle, p = p_rot, seed = random_seed)
                    image, bboxes = AugmentImage.shear(image, bboxes, shear_angle = shear_angle, p = p_shear, seed = random_seed)
                    image = AugmentImage.posterize(image, p = p_post, seed = random_seed)
                    image = AugmentImage.greyscale(image, p = p_grey, seed = random_seed)
                    image = AugmentImage.blur(image, p = p_blur, seed = random_seed)
                    image = AugmentImage.clahe(image, p = p_clahe, seed = random_seed)
                    image = AugmentImage.cutout(image, p = p_cutout, seed = random_seed)
                    image = AugmentImage.hsv_augment(image, hgain = hgain, sgain = sgain, vgain = vgain, p = p_hsv, seed = random_seed)
                    image = AugmentImage.shuffle_channel(image, p = p_shuffle, seed = random_seed)
                    
                    if j == 0:
                        img_lst1.append(np.array(image))
                        bboxes_lst1.append(bboxes)
                    if j == 1:
                        img_lst2.append(np.array(image))
                        bboxes_lst2.append(bboxes)
                    if j == 2:
                        img_lst3.append(np.array(image))
                        bboxes_lst3.append(bboxes)
                    if j == 3:
                        img_lst4.append(np.array(image))
                        bboxes_lst4.append(bboxes)
                    if j == 4:
                        img_lst5.append(np.array(image))
                        bboxes_lst5.append(bboxes)
                    if j == 5:
                        img_lst6.append(np.array(image))
                        bboxes_lst6.append(bboxes)
                    if j == 6:
                        img_lst7.append(np.array(image))
                        bboxes_lst7.append(bboxes)
                    if j == 7:
                        img_lst8.append(np.array(image))
                        bboxes_lst8.append(bboxes)
                    if j == 8:
                        img_lst9.append(np.array(image))
                        bboxes_lst9.append(bboxes)

            # ensure img_lists have same length
            img_lst1 = img_lst1[:seq_len] if len(img_lst1) > seq_len else img_lst1
            img_lst2 = img_lst2[:seq_len] if len(img_lst2) > seq_len else img_lst2
            img_lst3 = img_lst3[:seq_len] if len(img_lst3) > seq_len else img_lst3
            img_lst4 = img_lst4[:seq_len] if len(img_lst4) > seq_len else img_lst4
            img_lst5 = img_lst5[:seq_len] if len(img_lst5) > seq_len else img_lst5
            img_lst6 = img_lst6[:seq_len] if len(img_lst6) > seq_len else img_lst6
            img_lst7 = img_lst7[:seq_len] if len(img_lst7) > seq_len else img_lst7
            img_lst8 = img_lst8[:seq_len] if len(img_lst8) > seq_len else img_lst8
            img_lst9 = img_lst9[:seq_len] if len(img_lst9) > seq_len else img_lst9

            # ensure bboxes have same length
            bboxes_lst1 = bboxes_lst1[:seq_len] if len(bboxes_lst1) > seq_len else bboxes_lst1
            bboxes_lst2 = bboxes_lst2[:seq_len] if len(bboxes_lst2) > seq_len else bboxes_lst2
            bboxes_lst3 = bboxes_lst3[:seq_len] if len(bboxes_lst3) > seq_len else bboxes_lst3
            bboxes_lst4 = bboxes_lst4[:seq_len] if len(bboxes_lst4) > seq_len else bboxes_lst4
            bboxes_lst5 = bboxes_lst5[:seq_len] if len(bboxes_lst5) > seq_len else bboxes_lst5
            bboxes_lst6 = bboxes_lst6[:seq_len] if len(bboxes_lst6) > seq_len else bboxes_lst6
            bboxes_lst7 = bboxes_lst7[:seq_len] if len(bboxes_lst7) > seq_len else bboxes_lst7
            bboxes_lst8 = bboxes_lst8[:seq_len] if len(bboxes_lst8) > seq_len else bboxes_lst8
            bboxes_lst9 = bboxes_lst9[:seq_len] if len(bboxes_lst9) > seq_len else bboxes_lst9

            # repeat from beginning if sequence is smaller than intended sequence len
            for img_lst in [img_lst1, img_lst2, img_lst3, img_lst4, img_lst5, img_lst6, img_lst7, img_lst8, img_lst9]:
                while len(img_lst) < seq_len:
                    imgs_to_add = seq_len - len(img_lst)
                    img_lst.extend(img_lst[:imgs_to_add])

            for bboxes_lst in [bboxes_lst1, bboxes_lst2, bboxes_lst3, bboxes_lst4, bboxes_lst5, bboxes_lst6, bboxes_lst7, bboxes_lst8, bboxes_lst9]:
                while len(bboxes_lst) < seq_len:
                    bboxes_to_add = seq_len - len(bboxes_lst)
                    bboxes_lst.extend(bboxes_lst[:bboxes_to_add])

            # store mosaiced img and bboxes as list to create sequence
            for k in range(len(img_lst1)):

                mosaic_img = np.vstack([np.hstack([img_lst1[k], img_lst2[k], img_lst3[k]]),
                                   np.hstack([img_lst4[k], img_lst5[k], img_lst6[k]]),
                                   np.hstack([img_lst7[k], img_lst8[k], img_lst9[k]])])

                # 1.top left image of mosaic: no adjustment so skip
                # note: height is the height of a single mosaic tile not the height of the total image
                # 2.top middle image of mosaic: (x1,y1,x2,y2) + (height, 0, height, 0)
                # 3.top right image of mosaic: (x1,y1,x2,y2) + (height*2, 0, height*2)
                # 4.middle left image of mosaic: (x1,y1,x2,y2) + (0, height, 0, height)
                # 5.middle middle image of mosaic: (x1,y1,x2,y2) + (height, height, height, height)
                # 6.middle right image of mosaic: (x1,y1,x2,y2) + (height*2, height, height*2, height)
                # 7.bottom left image of mosaic: (x1,y1,x2,y2) + (0, height*2, 0, height*2)
                # 8.bottom middle  image of mosaic: (x1,y1,x2,y2) + (height, height*2, height, height*2)
                # 9.bottom right image of mosaic: (x1,y1,x2,y2) + (height*2, height*2, height*2, height*2)
                # 10. we add an additional 0 at the end to not change the class 
                mosaic_bbox = np.vstack([bboxes_lst1[k], 
                                         bboxes_lst2[k] + [height , 0, height, 0, 0], 
                                         bboxes_lst3[k] + [height*2, 0, height*2, 0, 0], 
                                         bboxes_lst4[k] + [0, height, 0, height, 0],
                                         bboxes_lst5[k] + [height, height, height, height, 0],
                                         bboxes_lst6[k] + [height*2, height, height*2, height, 0],
                                         bboxes_lst7[k] + [0, height*2, 0, height*2, 0],
                                         bboxes_lst8[k] + [height, height*2, height, height*2, 0],
                                         bboxes_lst9[k] + [height*2, height*2, height*2, height*2, 0]])
                
                # base augmentations to ensure consistency, normalization and yolo format
                mosaic_img = AugmentImage.resize(mosaic_img, img_size)
                # convert brg to rgb
                mosaic_img = cv.cvtColor(mosaic_img, cv.COLOR_BGR2RGB)
                # normalise # consistent with ImageNet backbone
                mosaic_img = AugmentImage.normalize(mosaic_img) #, mean = [0.485, 0.456, 0.406], std = [0.229, 0.244, 0.255])
                #image = Image.fromarray(image)

                # convert opencv pascalvoc format to yolo
                mosaic_bbox = AugmentBoundingBox.opencv_to_yolo(mosaic_bbox, img_size, img_size)
                final_images.append(mosaic_img)
                final_bboxes.append(mosaic_bbox)

            return final_images, final_bboxes
        
    def mixup_seq(img_size, img_dir,
            label_dir,
            vid_id_csv, 
            frame_ids_dir,
            p_scale = 1.0, scale_factor = 0.9,
            p_trans = 1.0, translate_factor = 0.1,
            p_rot = 0.3, rotation_angle = 45.0,
            p_shear = 0.0, shear_angle = 10.0,
            p_hflip = 0.5,
            p_hsv = 1.0, hgain = 0.015, sgain = 0.7, vgain = 0.4,
            p_grey = 0.1,
            p_blur = 0.1,
            p_clahe = 0.1,
            p_cutout = 0.0,
            p_shuffle = 0.1,
            p_post = 0.1,
            seq_len = 2):
            
        all_vid_frame_ids = []
        all_vid_frame_ids_paths = []
        all_sample_rates = []
        all_vid_folder_ids = []

        for i in range(2):
            random_index = random.randint(0, len(vid_id_csv) - 1)
            vid_folder_id = vid_id_csv.iloc[random_index, 0] #random.choice([line.strip() for line in open(vid_id_csv)])
            #print("Mixup vid_folder_id:", vid_folder_id)
            #print("Mixup frame_ids_dir:", frame_ids_dir)
            vid_frame_ids_path = frame_ids_dir + vid_folder_id
            vid_frame_ids = pd.read_csv(f'{vid_frame_ids_path}.csv')
            sampling_rate = sampling_rate = max(1, len(vid_frame_ids) // seq_len)
            all_vid_frame_ids.append(vid_frame_ids)
            all_vid_frame_ids_paths.append(vid_frame_ids_path)
            all_sample_rates.append(sampling_rate)
            all_vid_folder_ids.append(vid_folder_id)

        img_lst1, img_lst2 = [], []
        bboxes_lst1, bboxes_lst2 = [], []
        final_images = []
        final_bboxes = []

        for j in range(0, len(all_vid_frame_ids)):
            # random seed to ensure consistent augmentation across sequence
            random_seed = np.random.randint(0, 10000)
            for i in range(0, len(all_vid_frame_ids[j]), all_sample_rates[j]):
                label_path = os.path.join(label_dir, all_vid_folder_ids[j], all_vid_frame_ids[j].iloc[i, 0]).replace("\\", "/")
                img_path = os.path.join(img_dir, all_vid_folder_ids[j], all_vid_frame_ids[j].iloc[i, 0]).replace("\\", "/").replace(".txt", ".JPEG")
                bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).astype(np.float32)
                    
                image = np.array(Image.open(img_path).convert("RGB"))
                image = AugmentImage.rgb_to_bgr(image)
                # resize image
                image = AugmentImage.resize(image, img_size)
                height, width = image.shape[:2]
                # conver to opencv format for augmentations
                bboxes = AugmentBoundingBox.yolo_to_opencv(bboxes, width, height)
                # scale
                image, bboxes = AugmentImage.scale(image, bboxes, scale_factor = scale_factor, p = p_scale, seed = random_seed)
                # translate
                image, bboxes = AugmentImage.translate(image, bboxes, translate_factor = translate_factor, p = p_trans, seed = random_seed)
                # horizontal flip image
                image, bboxes = AugmentImage.horizontal_flip(image, bboxes, p = p_hflip, seed = random_seed)
                image, bboxes = AugmentImage.rotate(image, bboxes, rotation_angle = rotation_angle, p = p_rot, seed = random_seed)
                image, bboxes = AugmentImage.shear(image, bboxes, shear_angle = shear_angle, p = p_shear, seed = random_seed)
                image = AugmentImage.posterize(image, p = p_post, seed = random_seed)
                image = AugmentImage.greyscale(image, p = p_grey, seed = random_seed)
                image = AugmentImage.blur(image, p = p_blur, seed = random_seed)
                image = AugmentImage.clahe(image, p = p_clahe, seed = random_seed)
                image = AugmentImage.cutout(image, p = p_cutout, seed = random_seed)
                image = AugmentImage.hsv_augment(image, hgain = hgain, sgain = sgain, vgain = vgain, p = p_hsv, seed = random_seed)
                image = AugmentImage.shuffle_channel(image, p = p_shuffle, seed = random_seed)
                    
                if j == 0:
                    img_lst1.append(np.array(image))
                    bboxes_lst1.append(bboxes)

                if j == 1:
                    img_lst2.append(np.array(image))
                    bboxes_lst2.append(bboxes)

        # ensure img_lists have same length
        img_lst1 = img_lst1[:seq_len] if len(img_lst1) > seq_len else img_lst1
        img_lst2 = img_lst2[:seq_len] if len(img_lst2) > seq_len else img_lst2

        # ensure bboxes have same length
        bboxes_lst1 = bboxes_lst1[:seq_len] if len(bboxes_lst1) > seq_len else bboxes_lst1
        bboxes_lst2 = bboxes_lst2[:seq_len] if len(bboxes_lst2) > seq_len else bboxes_lst2

        # repeat from beginning if sequence is smaller than intended sequence len
        for img_lst in [img_lst1, img_lst2]:
            while len(img_lst) < seq_len:
                imgs_to_add = seq_len - len(img_lst)
                img_lst.extend(img_lst[:imgs_to_add])

        for bboxes_lst in [bboxes_lst1, bboxes_lst2]:
            while len(bboxes_lst) < seq_len:
                bboxes_to_add = seq_len - len(bboxes_lst)
                bboxes_lst.extend(bboxes_lst[:bboxes_to_add])
                        
        for k in range(len(img_lst1)):
            lambda_mix = np.random.beta(8.0, 8.0)
            mixedup_img = (img_lst1[k] * lambda_mix + img_lst2[k] * (1 - lambda_mix)).astype(np.uint8)
            mixedup_bbox = np.vstack([bboxes_lst1[k], bboxes_lst2[k]])
            # base augmentations to ensure consistency, normalization and yolo format
            mixedup_img = AugmentImage.resize(mixedup_img, img_size)
            # convert brg to rgb
            mixedup_img = cv.cvtColor(mixedup_img, cv.COLOR_BGR2RGB)
            # normalise # consistent with ImageNet backbone
            mixedup_img = AugmentImage.normalize(mixedup_img) #, mean = [0.485, 0.456, 0.406], std = [0.229, 0.244, 0.255])
            #image = Image.fromarray(image)

            # convert opencv pascalvoc format to yolo
            mixedup_bbox = AugmentBoundingBox.opencv_to_yolo(mixedup_bbox, img_size, img_size)
            final_images.append(mixedup_img)
            final_bboxes.append(mixedup_bbox)

        return final_images, final_bboxes
            

class AugmentBoundingBox:
    @staticmethod
    def compute_area(bboxes):
        area = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:,3] - bboxes[:,1])
        return area

    def clip(bboxes, img_size, min_visibility = 0.4, width_height_threshold = 4, aspect_ratio_threshold = 20):
        eps = 1e-9
        original_area = AugmentBoundingBox.compute_area(bboxes)
        # clips bounding boxes to be within the image
        # cliped below at 1 and above at image -1  assuming
        # a square image of size image size by image size
        bboxes[:, :4] = np.clip(bboxes[:, :4], 1, img_size - 1, dtype=np.float32)

        new_area = AugmentBoundingBox.compute_area(bboxes)
        area_dif = (original_area - new_area) / (new_area + eps)

        # Compute width and height of bounding boxes
        widths = bboxes[:, 2] - bboxes[:, 0]
        heights = bboxes[:, 3] - bboxes[:, 1]
        aspect_ratios = np.maximum(widths / (heights + eps), heights / (widths + eps))

        # Mask out bounding boxes based on width and height threshold, min visbibility and aspect ratio
        mask = ( (widths > width_height_threshold) &
                    (heights > width_height_threshold) &
                    (aspect_ratios < aspect_ratio_threshold) &
                    (area_dif < (1 - min_visibility))
                    ).astype(int)

        bboxes = bboxes[mask == 1,:]

        return bboxes

    def compute_corners(bboxes):
        c = (bboxes[:,4])[:, None]
        # width: bottom corner right x2 - top left corner x1
        width = (bboxes[:,2] - bboxes[:,0])[:, None]
        # height: bottom
        height = (bboxes[:,3] - bboxes[:,1])[:, None]
        x1 = (bboxes[:,0])[:, None]
        y1 = (bboxes[:,1])[:, None]
        # compute top right corner values x2, y2
        x2 = x1 + width
        y2 = y1
        # compute bottom left corner values x3, y3
        x3 = x1
        y3 = y1 + height
        x4 = (bboxes[:,2])[:, None]
        y4 = (bboxes[:,3])[:, None]
        corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4,c))
        return corners

    def corner_to_opencv(corners):
        c = (corners[:,8])[:,None]
        x_corner_coords = corners[:,[0,2,4,6]]
        y_corner_coords = corners[:,[1,3,5,7]]
        x1 = np.min(x_corner_coords,1)[:, None]
        y1 = np.min(y_corner_coords,1)[:, None]
        x2 = np.max(x_corner_coords,1)[:, None]
        y2 = np.max(y_corner_coords,1)[:, None]
        return np.hstack((x1, y1, x2, y2, c))

    def opencv_to_yolo(bboxes, width, height):
        """
        Convert pascal voc or opencv bounding box coordinates to YOLO format.
        Opencv format are two points: 
            - point 1 = (topleft corner x, topleft corner y) = x1, y1
            - point 2 = (bottom right corner x, bottom right corner y) = x2, y2

        Arguments:
        - boxes: np.array of shape N,4: N number of boxes and 4 = point1(x1, y1), point2(x2, y2)
        - width: Width of the image.
        - height: Height of the image.

        Returns:
        Yolo format bounding box centerx, centery, width, height of bounding box.
        """
        if bboxes.ndim == 1:
            #print("Empty bounding box:", bboxes)
            # if no bboxes then return bboxes and handle exception in dataset.py
            return bboxes
        c = bboxes[:,4]
        x1, y1, x2, y2 = bboxes[:,0], bboxes[:,1], bboxes[:,2], bboxes[:,3]
        x = (x2 + x1) / (2 * width)
        y = (y2 + y1) / (2 * height)
        w = (x2 - x1) / width
        h = (y2 - y1) / height
        # return yolo format: xcenter, ycenter, width, height normalised
        return np.round(np.column_stack((x, y, w, h, c)).astype(np.float32), 8)

    def yolo_to_opencv(bboxes, width, height):
        """
        Convert YOLO-format bounding box coordinates to OpenCV-format.

        Arguments:
        - boxes: np.array of shape N,4: N number of boxes and 4 x, y, w, h Yolo format
        - width: Width of the image in pixels.
        - height: Height of the image in pixels.

        Returns:
        x1, y1, x2, y2: Coordinates of the top-right and bottom-left corners of the bounding box in OpenCV format.
        """
        c = bboxes[:,4]
        x, y, w, h = bboxes[:,0], bboxes[:,1], bboxes[:,2], bboxes[:,3]
        w = w * width
        h = h * height
        x1 = ((2 * x * width) - w) / 2
        y1 = ((2 * y * height) - h) / 2
        x2 = x1 + w
        y2 = y1 + h
        return np.column_stack((x1, y1, x2, y2, c))
