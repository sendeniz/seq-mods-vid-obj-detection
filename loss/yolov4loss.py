import random
import torch
import torch.nn as nn
from utils.utils import box_intersection_over_union

class YoloV4Loss(nn.Module):
    def __init__(self, lambda_class = 1, lambda_noobj = 4, lambda_obj = 6, lambda_box = 8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = lambda_class #1
        self.lambda_noobj = lambda_noobj #10
        self.lambda_obj = lambda_obj #6 #2
        self.lambda_box = lambda_box #8 #10

    def forward(self, preds, target, anchors, mode ='ciou'):
        # Check where obj and noobj (we ignore if target == -1)
        obj = target[..., 0] == 1  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i
        # no object loss
        no_object_loss = self.bce(
            (preds[..., 0:1][noobj]), (target[..., 0:1][noobj]),
        )

        if torch.sum(obj) == 0:
            box_loss_val = 0
            obj_loss_val = 0
            noobj_loss_val = self.lambda_noobj * no_object_loss
            class_loss_val = 0
            yolov4_loss_val =  (box_loss_val + obj_loss_val + noobj_loss_val + class_loss_val)
            return yolov4_loss_val

        # object loss
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat([self.sigmoid(preds[..., 1:3]), torch.exp(preds[..., 3:5]) * anchors], dim=-1)
        ious = box_intersection_over_union(box_preds[obj], target[..., 1:5][obj], mode = mode).detach()
        object_loss = self.mse(self.sigmoid(preds[..., 0:1][obj]), ious * target[..., 0:1][obj])

        #  bounding box coordinate loss
        preds[..., 1:3] = self.sigmoid(preds[..., 1:3])  # x,y coordinates
        target[..., 3:5] = torch.log( (1e-16 + target[..., 3:5] / anchors) )  # width, height coordinates
        box_loss = self.mse(preds[..., 1:5][obj], target[..., 1:5][obj])

        # class loss
        class_loss = self.entropy(
            (preds[..., 5:][obj]), (target[..., 5][obj].long()),
        )
        box_loss_val = self.lambda_box * box_loss
        obj_loss_val = self.lambda_obj * object_loss
        noobj_loss_val = self.lambda_noobj * no_object_loss
        class_loss_val = self.lambda_class * class_loss
        yolov4_loss_val =  (box_loss_val + obj_loss_val + noobj_loss_val + class_loss_val)

        return yolov4_loss_val


class YoloV4Loss2(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1 #1
        self.lambda_noobj = 4 #10
        self.lambda_obj = 2 #6 #2
        self.lambda_box = 8 #8 #10

    def forward(self, preds, target, anchors, mode ='iou'):
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
            yolov4_loss_val = (box_loss_val + obj_loss_val + noobj_loss_val + class_loss_val)
            return yolov4_loss_val

        # object loss
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        xy_offset = self.sigmoid(preds[..., 1:3])
        wh_cell = torch.exp(preds[..., 3:5])*anchors
        pred_bboxes = torch.cat([xy_offset, wh_cell], dim=-1)
        # gt boxes
        xy_offset = target[..., 1:3]
        wh_cell = target[..., 3:5]
        true_bboxes = torch.cat([xy_offset, wh_cell], dim=-1)
        # compute iou
        iou = box_intersection_over_union(pred_bboxes[obj_mask], true_bboxes[obj_mask], mode = mode) #.detach()
        # compute objectness loss
        object_loss = self.bce(preds[..., 0:1][obj_mask], target[..., 0:1][obj_mask]*iou.detach().clamp(0))
        #  bounding box coordinate loss
        # x,y coordinates
        preds[..., 1:3] = self.sigmoid(preds[..., 1:3])
        # width, height coordinates
        target[..., 3:5] = torch.log( (1e-16 + target[..., 3:5] / anchors) )  
        target_wh = torch.log(1e-16 + target[..., 3:5] / anchors)
        box_loss = self.bce(preds[..., 1:3][obj_mask],
                            target[..., 1:3][obj_mask])
        #box_loss += self.mse(preds[..., 3:5][obj_mask],
        #            target_wh[obj_mask])
        box_loss += (1-iou).mean()

        # class loss
        class_loss = self.entropy(
            (preds[..., 5:][obj_mask]), (target[..., 5][obj_mask].long()),
        )
        box_loss_val = self.lambda_box * box_loss
        obj_loss_val = self.lambda_obj * object_loss
        noobj_loss_val = self.lambda_noobj * no_object_loss
        class_loss_val = self.lambda_class * class_loss
        yolov4_loss_val = (box_loss_val + obj_loss_val + noobj_loss_val + class_loss_val)

        return yolov4_loss_val