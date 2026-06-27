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
from tqdm import tqdm  # Add progress bar for pre-loading
cv.setNumThreads(1)

img_size = 640 

if img_size == 640:
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

if img_size == 608:
    anchors = [
  [(0.23, 0.18), (0.32, 0.40), (0.75, 0.66)],
  [(0.06, 0.12), (0.12, 0.09), (0.12, 0.24)],
  [(0.02, 0.03), (0.03, 0.06), (0.07, 0.05)], 
  ]
    S = [19, 38, 76]

if img_size == 416:
    anchors = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]
    S = [13, 26, 52]

class CoCoDataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_dir,
        label_dir,
        img_size=640,  # Default to 640
        C=80,
        mode="test",
        preload_to_memory=True,  # Option to preload data
    ):
        self.annotations = pd.read_csv(csv_file)
        self.csv_file = csv_file
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.mode = mode
        self.C = C
        self.ignore_iou_thresh = 0.5
        self.preload_to_memory = preload_to_memory
        
        self.all_image_paths = self.annotations.iloc[:, 0].tolist()
        self.all_label_paths = self.annotations.iloc[:, 1].tolist()

        # Pre-load all data into memory during initialization
        if self.preload_to_memory:
            print(f"Pre-loading {len(self.annotations)} samples into memory...")
            self.images = []
            self.bboxes_list = []
            
            for i in tqdm(range(len(self.annotations))):
                label_path = os.path.join(self.label_dir, self.all_label_paths[i])
                img_path = os.path.join(self.img_dir, self.all_image_paths[i])
                
                # Load once during initialization
                bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).astype(np.float32)
                image = np.array(Image.open(img_path).convert("RGB"))
                
                self.images.append(image)
                self.bboxes_list.append(bboxes)
            print("Dataset pre-loaded into memory!")
        else:
            # Keep the old behavior (slow)
            self.images = None
            self.bboxes_list = None

        # Auto-configure anchors and grid sizes based on image_size
        self.anchors, self.S = self._get_anchors_and_scales(img_size)
        
        # Convert anchors to tensor (shape: [9, 2] for 3 scales)
        self.anchors = torch.tensor(
            [anchor for scale in self.anchors for anchor in scale]
        )
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = self.num_anchors // 3

    @staticmethod
    def _get_anchors_and_scales(img_size):
        """Returns anchors and grid sizes (S) for a given image_size."""
        if img_size == 640:
            anchors = [
                [(0.22, 0.17), (0.30, 0.38), (0.72, 0.63)],  # Large objects (20x20)
                [(0.06, 0.12), (0.12, 0.09), (0.11, 0.23)],  # Medium (40x40)
                [(0.02, 0.03), (0.03, 0.06), (0.06, 0.04)],  # Small (80x80)
            ]
            S = [20, 40, 80]  # 640/32=20, 640/16=40, 640/8=80

        elif img_size == 608:
            anchors = [
                [(0.23, 0.18), (0.32, 0.40), (0.75, 0.66)],
                [(0.06, 0.12), (0.12, 0.09), (0.12, 0.24)],
                [(0.02, 0.03), (0.03, 0.06), (0.07, 0.05)],
            ]
            S = [19, 38, 76]  # 608/32≈19, 608/16=38, 608/8=76

        elif img_size == 416:
            anchors = [
                [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
                [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
                [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
            ]
            S = [13, 26, 52]  # 416/32=13, 416/16=26, 416/8=52

        else:
            raise ValueError(f"Unsupported image_size: {img_size}")

        return anchors, S

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        # Get data from memory instead of disk
        if self.preload_to_memory:
            image = self.images[index].copy()  # Use copy to avoid modifying original
            bboxes = self.bboxes_list[index].copy()
        else:
            # Old slow method multiple read operations from drive
            label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
            bboxes = np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1).astype(np.float32)

            img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
            image = np.array(Image.open(img_path).convert("RGB"))
        
        if self.mode == 'test':
            image, bboxes = augment_data(image, bboxes, img_size=self.img_size, mode=self.mode)
        
        if self.mode == 'train':
            # Pass pre-loaded data for mosaic
            if self.preload_to_memory:
                image, bboxes = augment_data(image, bboxes, img_size=self.img_size,
                                             p_scale=1.0, scale_factor=0.9,
                                             p_trans=1.0, translate_factor=0.1,
                                             p_rot=0.3, rotation_angle=45.0,
                                             p_shear=0.3, shear_angle=10.0,
                                             p_hflip=0.5, 
                                             p_vflip=0.0, 
                                             p_mixup=0.3, 
                                             p_mosaic=0.3,  # set to 0.0 for testing
                                             p_hsv=0.3, hgain=0.1, sgain=0.9, vgain=0.9, 
                                             p_grey=0.1,
                                             p_blur=0.1, 
                                             p_clahe=0.1,  
                                             p_cutout=0.0, 
                                             p_shuffle=0.1,
                                             p_post=0.1, mode=self.mode,
                                             all_images=self.images,  # Pass all pre-loaded images
                                             all_bboxes=self.bboxes_list,  #Pass all pre-loaded bboxes
                                             img_dir=self.img_dir,
                                             label_dir=self.label_dir)
            else:
                # Old method
                image, bboxes = augment_data(image, bboxes, img_size=self.img_size,
                                             p_scale=1.0, scale_factor=0.9,
                                             p_trans=1.0, translate_factor=0.1,
                                             p_rot=0.3, rotation_angle=45.0,
                                             p_shear=0.3, shear_angle=10.0,
                                             p_hflip=0.5, 
                                             p_vflip=0.0, 
                                             p_mixup=0.3, 
                                             p_mosaic=0.3,  # TEMPORARILY DISABLE: set to 0.0 for testing
                                             p_hsv=0.3, hgain=0.1, sgain=0.9, vgain=0.9, 
                                             p_grey=0.1,
                                             p_blur=0.1, 
                                             p_clahe=0.1,  
                                             p_cutout=0.0, 
                                             p_shuffle=0.1,
                                             p_post=0.1, mode=self.mode,
                                             all_image_paths=self.all_image_paths,
                                             all_label_paths=self.all_label_paths,
                                             annotations_csv=self.csv_file,
                                             img_dir=self.img_dir,
                                             label_dir=self.label_dir)
    
        image = np.array(image)
        # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
        targets = [torch.zeros((self.num_anchors // 3, S, S, 6)) for S in self.S]
      
        for box in bboxes:
            iou_anchors = iou_width_height(torch.tensor(box[2:4]), self.anchors)
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

                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

        return image.astype(np.float16), tuple(targets)

def test(anchors=anchors, mode='train'):
    dataset = CoCoDataset("data/coco/test_10examples.csv", "data/coco/images/", "data/coco/labels/",
                         mode=mode)
    
    scaled_anchors = torch.tensor(anchors) / (1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2))
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    
    for idx, (x, y) in enumerate(loader):
        boxes = []

        for i in range(y[0].shape[1]):
            anchor = scaled_anchors[i]
            boxes += cells_to_bboxes(y[i], is_preds=False, S=y[i].shape[2], anchors=anchor)[0]
        
        boxes = non_max_suppression(boxes, iou_threshold=1.0, confidence_threshold=0.7)
        img = draw_bounding_box(x[0].permute(0, 1, 2).to("cpu") * 255, boxes)
        filename = f'figures/yolo_data_{idx}.png' 
        plt.imsave(filename, img)

#test()