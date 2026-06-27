import torch
import os
import cv2 as cv
import pandas as pd
from PIL import Image, ImageFile
from utils.utils import iou_width_height as iou
from utils.utils import non_max_suppression
from utils.utils import cells_to_bboxes
from utils.utils import draw_bounding_box_vid
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import warnings
from utils.transforms import augment_data
import random
from utils.augmentations import AugmentImage
cv.setNumThreads(1)

image_size = 608

if image_size == 608:
    anchors = [
        [(0.23, 0.18), (0.32, 0.40), (0.75, 0.66)],
        [(0.06, 0.12), (0.12, 0.09), (0.12, 0.24)],
        [(0.02, 0.03), (0.03, 0.06), (0.07, 0.05)],
    ]
    S = [19, 38, 76]

if image_size == 416:
    anchors = [
        [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
        [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
        [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
    ]
    S = [13, 26, 52]


class ImageNetVidDataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        vid_id_csv,
        frame_ids_dir,
        image_size=image_size,
        S=S,
        C=30,
        mode="train",
        seq_len=32,
    ):
        self.annotations = pd.read_csv(csv_file)
        self.mode = mode
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.S = S
        self.anchors = torch.tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3
        self.C = C
        self.ignore_iou_thresh = 0.5
        self.seq_len = seq_len
        self.vid_id_csv = pd.read_csv(vid_id_csv)
        self.frame_ids_dir = frame_ids_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        bboxes_lst = []
        imgs_lst = []
        targets_lst = []
        vid_folder_id = self.vid_id_csv.iloc[index, 0] #random.choice([line.strip() for line in open(self.vid_id_txt)])
        vid_frame_ids_path = self.frame_ids_dir + vid_folder_id
        vid_frame_ids = pd.read_csv(f"{vid_frame_ids_path}.csv")
        # ensure esampleing rate is 1
        sampling_rate = max(1, len(vid_frame_ids) // self.seq_len)

        # generate a random seed that is passed to the augmentation to ensure
        # that each image within a sequence has the same augmentation applied
        random_seed = np.random.randint(0, 10000)

        for i in range(0, len(vid_frame_ids), sampling_rate):
            label_path = os.path.join(
                self.label_dir, vid_folder_id, vid_frame_ids.iloc[i, 0]
            ).replace("\\", "/")
            img_path = (
                os.path.join(self.img_dir, vid_folder_id, vid_frame_ids.iloc[i, 0])
                .replace("\\", "/")
                .replace(".txt", ".JPEG")
            )

            bboxes = np.roll(
                np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1
            ).astype(np.float32)
            image = np.array(Image.open(img_path).convert("RGB"))
            imgs_lst.append(np.array(image))
            bboxes_lst.append(bboxes)

            if self.mode == "test":
                imgs_lst[i // sampling_rate], bboxes_lst[i // sampling_rate] = (
                    augment_data(image, bboxes, image_size=image_size, mode=self.mode)
                )

            if self.mode == "train":
                imgs_lst[i // sampling_rate], bboxes_lst[i // sampling_rate] = (
                    augment_data(
                        image,
                        bboxes,
                        image_size=image_size,
                        p_scale=1.0,
                        scale_factor=0.9,
                        p_trans=1.0,
                        translate_factor=0.1,
                        p_rot=0.0,
                        rotation_angle=45.0,
                        p_shear=0.0,
                        shear_angle=10.0,
                        p_hflip=0.0,
                        p_vflip=0.0,
                        p_mixup=0.0,
                        p_mosaic=0.0,
                        p_hsv=1.0,
                        hgain=0.015,
                        sgain=0.7,
                        vgain=0.4,
                        p_grey=0.1,
                        p_blur=0.1,
                        p_clahe=0.1,
                        p_cutout=0.0,
                        p_shuffle=0.1,
                        p_post=0.1,
                        mode=self.mode,
                        seed=random_seed,
                    )
                )

        # overwrites regular train augmentation for mosaic or mixup
        if self.mode == "train" and np.random.rand() <= 0.0:
            mosaic_or_mixup = np.random.choice(["mosaic", "mixup"])
            if mosaic_or_mixup == "mosaic":
                imgs_lst, bboxes_lst = AugmentImage.mosaic_seq(
                    image_size=image_size,
                    img_dir=self.img_dir,
                    label_dir=self.label_dir,
                    vid_id_csv=self.vid_id_csv,
                    frame_ids_dir=self.frame_ids_dir,
                    seq_len=self.seq_len,
                )
            if mosaic_or_mixup == "mixup":
                imgs_lst, bboxes_lst = AugmentImage.mixup_seq(
                    image_size=image_size,
                    img_dir=self.img_dir,
                    label_dir=self.label_dir,
                    vid_id_csv=self.vid_id_csv,
                    frame_ids_dir=self.frame_ids_dir,
                    seq_len=self.seq_len,
                )
        # substract 1 from class category because ILSVCR counts at 1 and not 0
        bboxes_lst = [
            np.hstack((arr[:, :-1], np.expand_dims(arr[:, -1] - 1, axis=1)))
            for arr in bboxes_lst
        ]

        # if sequence is too long remove additional elements
        if len(imgs_lst) > self.seq_len:
            imgs_lst = imgs_lst[: self.seq_len]

        if len(bboxes_lst) > self.seq_len:
            bboxes_lst = bboxes_lst[: self.seq_len]

        # if sequence is too small repeat existing sequence until desired size is reached
        while len(imgs_lst) < self.seq_len:
            imgs_to_add = self.seq_len - len(imgs_lst)
            # repeat elements from the beginning until length is equal to seq len
            imgs_lst.extend(imgs_lst[:imgs_to_add])

        while len(bboxes_lst) < self.seq_len:
            bboxes_to_add = self.seq_len - len(bboxes_lst)
            # repeat elements from the beginning until length is equal to seq len
            bboxes_lst.extend(bboxes_lst[:bboxes_to_add])

        for seq_idx in range(0, len(imgs_lst)):
            bboxes = bboxes_lst[seq_idx]

            targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]

            for box in bboxes:
                iou_anchors = iou(
                    torch.tensor(box[2:4], dtype=torch.float32), self.anchors
                )

                anchor_indices = iou_anchors.argsort(descending=True, dim=0)

                x, y, width, height, class_label = box

                has_anchor = [False] * 3  # each scale should have one anchor

                for anchor_idx in anchor_indices:
                    scale_idx = anchor_idx // self.num_anchors_per_scale
                    anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                    S = self.S[scale_idx]
                    i, j = int(S * y), int(S * x)  # which cell
                    anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                    if not anchor_taken and not has_anchor[scale_idx]:
                        targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                        x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                        width_cell, height_cell = (
                            width * S,
                            height * S,
                        )  # can be greater than 1 since it's relative to cell
                        box_coordinates = torch.tensor(
                            [x_cell, y_cell, width_cell, height_cell]
                        )
                        targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                        targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                        has_anchor[scale_idx] = True

                    elif (
                        not anchor_taken
                        and iou_anchors[anchor_idx] > self.ignore_iou_thresh
                    ):
                        targets[scale_idx][ 
                            anchor_on_scale, i, j, 0
                        ] = -1  # ignore prediction

            targets_lst.append(tuple(targets))
        # cast images to float16 to save gpu memory
        imgs_lst = [img.astype(np.float16) for img in imgs_lst]
        

        return imgs_lst, targets_lst


def test(anchors=anchors, mode="train"):
    dataset = ImageNetVidDataset(
        csv_file="data/ILSVRC2015/Data/train10examples.csv",
        img_dir="data/ILSVRC2015/Data/VID/train/",
        label_dir="data/ILSVRC2015/Data/labels/VID/train/",
        vid_id_csv="data/ILSVRC2015/Data/labels/train_vid_id10examples.csv",
        frame_ids_dir="data/ILSVRC2015/Data/labels/train_frame_ids/",
        mode=mode,
        seq_len=16,
    )

    scaled_anchors = torch.tensor(anchors) / (
        1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    loader = DataLoader(dataset=dataset, batch_size=5, shuffle=False)

    for idx, (x, y) in enumerate(loader):
        for j in range(0, len(y)):
            boxes = []
            for i in range(y[j][0].shape[1]):
                anchor = scaled_anchors[i]
                boxes += cells_to_bboxes(
                    y[j][i], is_preds=False, S=y[j][i].shape[2], anchors=anchor
                )[0]

            boxes = non_max_suppression(
                boxes, iou_threshold=1.0, confidence_threshold=0.7
            )
            img = draw_bounding_box_vid(x[j][0].permute(0, 1, 2).to("cpu") * 255, boxes)
            filename = f"figures/yolo_data_{idx}_{j}.png"
            plt.imsave(filename, img)


#test()
