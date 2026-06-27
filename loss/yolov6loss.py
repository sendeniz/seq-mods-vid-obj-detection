import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import box_intersection_over_union

class VarifocalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy(pred_score.float(), gt_score.float(), reduction='none') * weight).sum()
        return loss

class DistributionFocalLoss(nn.Module):
    """Distribution Focal Loss (DFL) for bounding box regression"""
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max
        
    def forward(self, pred_dist, target):
        target_left = target.long()
        target_right = target_left + 1
        target_right = target_right.clamp(max=self.reg_max)
        
        weight_right = target - target_left
        weight_left = 1 - weight_right
        
        loss_left = F.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1),
            target_left.view(-1),
            reduction='none'
        ).view(target_left.shape) * weight_left
        
        loss_right = F.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1),
            target_right.view(-1), 
            reduction='none'
        ).view(target_left.shape) * weight_right
        
        return (loss_left + loss_right).mean(-1, keepdim=True)

class YoloV6Loss(nn.Module):
    def __init__(self, lambda_class=1, lambda_noobj=4, lambda_obj=2, lambda_box=8, reg_max=16):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.varifocal = VarifocalLoss()
        self.dfl = DistributionFocalLoss(reg_max=reg_max)
        self.sigmoid = nn.Sigmoid()
        self.reg_max = reg_max
        self.proj = nn.Parameter(torch.linspace(0, reg_max, reg_max + 1), requires_grad=False)

        # Loss weights
        self.lambda_class = lambda_class
        self.lambda_noobj = lambda_noobj
        self.lambda_obj = lambda_obj
        self.lambda_box = lambda_box

    def forward(self, preds, target, anchors, mode='siou'):
        # Check where obj and noobj (we ignore if target == -1)
        obj_mask = target[..., 0] == 1  # in paper this is Iobj_i
        noobj_mask = target[..., 0] == 0  # in paper this is Inoobj_i

        # No object loss
        no_object_loss = self.bce(
            (preds[..., 0:1][noobj_mask]), 
            (target[..., 0:1][noobj_mask]),
        )

        if torch.sum(obj_mask) == 0:
            box_loss_val = 0
            obj_loss_val = 0
            noobj_loss_val = self.lambda_noobj * no_object_loss
            class_loss_val = 0
            yolov6_loss_val = (box_loss_val + obj_loss_val + noobj_loss_val + class_loss_val)
            return yolov6_loss_val

        # Object loss
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        xy_offset = self.sigmoid(preds[..., 1:3])
        wh_cell = torch.exp(preds[..., 3:5]) * anchors
        pred_bboxes = torch.cat([xy_offset, wh_cell], dim=-1)
        
        # GT boxes
        xy_offset = target[..., 1:3]
        wh_cell = target[..., 3:5]
        true_bboxes = torch.cat([xy_offset, wh_cell], dim=-1)
        
        # Compute IoU
        iou = box_intersection_over_union(pred_bboxes[obj_mask], true_bboxes[obj_mask], mode=mode)
        
        # Compute objectness loss
        object_loss = self.bce(
            preds[..., 0:1][obj_mask], 
            target[..., 0:1][obj_mask] * iou.detach().clamp(0)
        )

        # Bounding box coordinate loss with DFL
        preds[..., 1:3] = self.sigmoid(preds[..., 1:3])  # x,y coordinates
        
        # Get predicted distribution for width/height
        pred_dist = preds[..., 6:6+4*(self.reg_max+1)].view(
            *preds[..., 6:6+4*(self.reg_max+1)].shape[:-1], 4, self.reg_max + 1
        )
        
        # Calculate target distances (normalized width/height)
        target_dist = target[..., 3:5] / anchors
        
        # Calculate DFL loss
        dfl_loss = self.dfl(
            pred_dist[obj_mask][..., 2:4],  # Only apply to width/height
            target_dist[obj_mask]
        )

        box_loss = self.bce(
            preds[..., 1:3][obj_mask],
            target[..., 1:3][obj_mask]
        )
        box_loss += dfl_loss.mean()  # Replace MSE with DFL
        box_loss += (1 - iou).mean()

        # Class loss (Varifocal implementation)
        gt_classes = target[..., 5][obj_mask].long()
        one_hot_label = F.one_hot(gt_classes, num_classes=preds[..., 5:].shape[-1]).float()
        pred_scores = self.sigmoid(preds[..., 5:][obj_mask])
        gt_scores = target[..., 0:1][obj_mask] * iou.detach().clamp(0)
        class_loss = self.varifocal(pred_scores, gt_scores, one_hot_label)

        # Weighted losses
        box_loss_val = self.lambda_box * box_loss
        obj_loss_val = self.lambda_obj * object_loss
        noobj_loss_val = self.lambda_noobj * no_object_loss
        class_loss_val = self.lambda_class * class_loss
        
        yolov6_loss_val = (box_loss_val + obj_loss_val + noobj_loss_val + class_loss_val)
        return yolov6_loss_val