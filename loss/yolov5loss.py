import random
import torch
import torch.nn as nn
from utils.utils import box_intersection_over_union

class YoloV5Loss(nn.Module):
    def __init__(self, lambda_class=0.5, lambda_noobj=1.0, lambda_obj=1.0, lambda_box=0.05):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # YOLOv5-style scaling factors
        self.lambda_class = lambda_class  # 0.5 (was 1)
        self.lambda_noobj = lambda_noobj  # 1.0 (was 4)  
        self.lambda_obj = lambda_obj      # 1.0 (was 2)
        self.lambda_box = lambda_box      # 0.05 (was 8 - 160x smaller!)

    def forward(self, preds, target, anchors, mode='ciou'):
        
        # Check where obj and noobj (we ignore if target == -1)
        obj_mask = target[..., 0] == 1  # in paper this is Iobj_i
        noobj_mask = target[..., 0] == 0  # in paper this is Inoobj_i
        
        # no object loss
        no_object_loss = self.bce(
            (preds[..., 0:1][noobj_mask]), (target[..., 0:1][noobj_mask]),
        )

        if torch.sum(obj_mask) == 0:
            box_loss_val = 0
            obj_loss_val = 0
            noobj_loss_val = self.lambda_noobj * no_object_loss
            class_loss_val = 0
            yolov5_loss_val = (box_loss_val + obj_loss_val + noobj_loss_val + class_loss_val)
            return yolov5_loss_val

        # object loss
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        
        # YOLOv5-style coordinate scaling
        xy_offset = self.sigmoid(preds[..., 1:3]) * 2 - 0.5  # Scale to -0.5 to 1.5
        wh_cell = (self.sigmoid(preds[..., 3:5]) * 2) ** 2 * anchors  # Scale to 0-4 * anchors
        
        pred_bboxes = torch.cat([xy_offset, wh_cell], dim=-1)
        
        # gt boxes
        xy_offset = target[..., 1:3]
        wh_cell = target[..., 3:5]
        true_bboxes = torch.cat([xy_offset, wh_cell], dim=-1)
        
        # compute iou
        iou = box_intersection_over_union(pred_bboxes[obj_mask], true_bboxes[obj_mask], mode=mode)
        
        object_loss = self.bce(
            preds[..., 0:1][obj_mask],  # Shape [421, 1]
            iou.detach().clamp(0)       # Shape [421, 1]
        )
        
        # bounding box coordinate loss ciou in yolov5
        box_loss = (1 - iou).mean()

        # class loss
        class_loss = self.entropy(
            (preds[..., 5:][obj_mask]), (target[..., 5][obj_mask].long()),
        )
        
        # Apply YOLOv5 scaling
        box_loss_val = self.lambda_box * box_loss
        obj_loss_val = self.lambda_obj * object_loss
        noobj_loss_val = self.lambda_noobj * no_object_loss
        class_loss_val = self.lambda_class * class_loss
        
        yolov5_loss_val = (box_loss_val + obj_loss_val + noobj_loss_val + class_loss_val)

        return yolov5_loss_val