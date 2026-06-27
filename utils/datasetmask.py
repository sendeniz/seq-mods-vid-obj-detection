import torch 
import os
import cv2 as cv
import pandas as pd
from PIL import Image, ImageFile
from utils.utils import iou_width_height
from utils.utils import non_max_suppression
from utils.utils import cells_to_bboxes
from utils.utils import draw_bounding_box
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from utils.transforms import augment_data
import warnings
cv.setNumThreads(1)

image_size = 640 

if image_size == 640:
    anchors = [
    # Large objects (assigned to smallest grid, e.g., 20x20)
    [(0.22, 0.17), (0.30, 0.38), (0.72, 0.63)],
    # Medium objects (e.g., 40x40)
    [(0.06, 0.12), (0.12, 0.09), (0.11, 0.23)],
    # Small objects (e.g., 80x80)
    [(0.02, 0.03), (0.03, 0.06), (0.06, 0.04)],
    ]
    # Grid sizes (640/32=20, 640/16=40, 640/8=80)
    S = [20, 40, 80]  

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

class CocoInstanceSeg(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        mask_dir,  # Directory containing mask folders
        image_size=608,
        C=80,
        mode="train",
        transform=None,  # Optional albumentations-style transform
    ):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.mode = mode
        self.C = C
        self.transform = transform
        self.ignore_iou_thresh = 0.5

        # Get anchors and grid sizes
        self.anchors, self.S = self._get_anchors_and_scales(image_size)
        self.anchors = torch.tensor([anchor for scale in self.anchors for anchor in scale])
        self.num_anchors_per_scale = len(self.anchors) // 3

    @staticmethod
    def _get_anchors_and_scales(image_size):
        """Returns anchors and grid sizes for given image size."""
        if image_size == 640:
            anchors = [
                [(0.22, 0.17), (0.30, 0.38), (0.72, 0.63)],  # Large (20x20)
                [(0.06, 0.12), (0.12, 0.09), (0.11, 0.23)],  # Medium (40x40)
                [(0.02, 0.03), (0.03, 0.06), (0.06, 0.04)],  # Small (80x80)
            ]
            S = [20, 40, 80]
        elif image_size == 608:
            anchors = [
                [(0.23, 0.18), (0.32, 0.40), (0.75, 0.66)],
                [(0.06, 0.12), (0.12, 0.09), (0.12, 0.24)],
                [(0.02, 0.03), (0.03, 0.06), (0.07, 0.05)],
            ]
            S = [19, 38, 76]
        elif image_size == 416:
            anchors = [
                [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
                [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
                [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
            ]
            S = [13, 26, 52]
        else:
            raise ValueError(f"Unsupported image_size: {image_size}")
        return anchors, S

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # Load image
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = np.array(Image.open(img_path).convert("RGB"))

        # Load bboxes
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        bboxes = np.roll(np.loadtxt(label_path, delimiter=" ", ndmin=2), 4, axis=1).astype(np.float32)

        # Load masks
        img_name = os.path.splitext(self.annotations.iloc[index, 0])[0]
        mask_folder = os.path.join(self.mask_dir, img_name)
        masks = []
        if os.path.exists(mask_folder):
            mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith(('.png', '.jpg'))])
            for mfile in mask_files:
                mask = np.array(Image.open(os.path.join(mask_folder, mfile)).convert("L")) / 255.0
                masks.append(mask)
        masks = np.stack(masks, axis=0) if masks else np.zeros((0, *image.shape[:2]))

        # Apply transformations
        if self.transform:
            transformed = self.transform(
                image=image,
                masks=masks,
                bboxes=bboxes
            )
            image = transformed["image"]
            masks = np.stack(transformed["masks"], axis=0) if len(transformed["masks"]) > 0 else masks
            bboxes = transformed["bboxes"]

        # Add mask indices to bboxes
        bboxes = [np.append(box, idx) for idx, box in enumerate(bboxes)]

        # Initialize targets [x,y,w,h,conf,class,mask_idx]
        targets = [torch.zeros((self.num_anchors_per_scale, S, S, 7)) for S in self.S]

        for box in bboxes:
            x, y, w, h, cls, mask_idx = box
            iou_anchors = iou_width_height(torch.tensor([w, h]), self.anchors)
            anchor_indices = iou_anchors.argsort(descending=True)
            
            has_anchor = [False] * 3
            for anchor_idx in anchor_indices:
                scale_idx = anchor_idx // self.num_anchors_per_scale
                anchor_on_scale = anchor_idx % self.num_anchors_per_scale
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x)
                
                if not targets[scale_idx][anchor_on_scale, i, j, 4] and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, :4] = torch.tensor([S*x-j, S*y-i, S*w, S*h])
                    targets[scale_idx][anchor_on_scale, i, j, 4] = 1  # obj confidence
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(cls)
                    targets[scale_idx][anchor_on_scale, i, j, 6] = int(mask_idx)
                    has_anchor[scale_idx] = True
                
                elif not targets[scale_idx][anchor_on_scale, i, j, 4] and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 4] = -1  # ignore

        return (
            torch.tensor(image).permute(2, 0, 1).float() / 255.0,  # [C,H,W] normalized
            torch.tensor(masks).float(),  # [N,H,W] binary masks
            tuple(targets)  # Multi-scale targets
        )

def test(anchors = anchors, mode = 'train'):

    dataset = CocoInstanceSeg("data/coco/train_10examples.csv", "data/coco/images/", "data/coco/labels/",
                          S = S, anchors = anchors, mode = mode)
    
    scaled_anchors = torch.tensor(anchors) / ( 1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2) )
    loader = DataLoader(dataset = dataset, batch_size = 1, shuffle = False)
    
    for idx, (x, y) in enumerate(loader):
        boxes = []

        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            boxes += cells_to_bboxes(y[i], is_preds=False, S=y[i].shape[2], anchors = anchor)[0]
        
        boxes = non_max_suppression(boxes, iou_threshold = 1.0, confidence_threshold = 0.7)
        #print(boxes)
        #print(x[0].shape)
        img = draw_bounding_box(x[0].permute(0, 1, 2).to("cpu") * 255, boxes)
        filename = f'figures/yolo_data_{idx}.png' 
        plt.imsave(filename, img)

#test()


