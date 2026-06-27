import torch
from torch import nn
import numpy as np
from collections import Counter
import cv2 as cv
import math
cv.setNumThreads(1)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def write_results(file_prefix, metrics, data_lists, run):
    for metric, data in zip(metrics, data_lists):
        with open(f'results/{file_prefix}_{metric}_run{run + 1}.txt', 'w') as values:
            values.write(str(data))

def iou_width_height(boxes1, boxes2):
    """
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union

def box_intersection_over_union(box1, box2, mode = 'iou'):
    """
    Calculates intersection of unions (IoU).
    Input: Boundbing box predictions (tensor) x, y, w, h (yolo format) of shape (N , 4)
            with N denoting the number of bounding boxes.
            Bounding box target/ground truth (tensor) x1, x2, y1, y2 of shape (N, 4).
            box format whether midpoint location or corner location of bounding boxes
            are used.
    Output: Intersection over union (tensor).
    """
    epsilon = 1e-9
    # Pred boxes
    box1_x1 = box1[..., 0:1] - (box1[..., 2:3] / 2)
    box1_y1 = box1[..., 1:2] - (box1[..., 3:4] / 2)
    box1_x2 = box1[..., 0:1] + (box1[..., 2:3] / 2)
    box1_y2 = box1[..., 1:2] + (box1[..., 3:4] / 2)
    box1_w = (box1_x2 - box1_x1).clamp(0)
    box1_h = (box1_y2 - box1_y1).clamp(0) + epsilon
    box1_area = (box1_w * box1_h) + epsilon
    # True boxes
    box2_x1 = box2[..., 0:1] - (box2[..., 2:3] / 2)
    box2_y1 = box2[..., 1:2] - (box2[..., 3:4] / 2)
    box2_x2 = box2[..., 0:1] + (box2[..., 2:3] / 2)
    box2_y2 = box2[..., 1:2] + (box2[..., 3:4] / 2)
    box2_w = (box2_x2 - box2_x1).clamp(0)
    box2_h = (box2_y2 - box2_y1).clamp(0) + epsilon
    box2_area = (box2_w * box2_h) + epsilon
    # Intersection boxes
    inter_x1 = torch.max(box1_x1, box2_x1)
    inter_y1 = torch.max(box1_y1, box2_y1)
    inter_x2 = torch.min(box1_x2, box2_x2)
    inter_y2 = torch.min(box1_y2, box2_y2)
    inter_w = (inter_x2 - inter_x1).clamp(0)
    inter_h = (inter_y2 - inter_y1).clamp(0)
    inter_area = (inter_w * inter_h) + epsilon
    union = (box1_area + box2_area - inter_area + epsilon)

    # Computer IoU
    iou = inter_area / union

    if mode == 'iou':
        return iou

    elif mode == 'giou':
        # Convex diagnal length
        convex_w = torch.max(box1_x2, box2_x2) - torch.min(box1_x1, box2_x1)
        convex_h = torch.max(box1_y2, box2_y2) - torch.min(box1_y1, box2_y1)
        convex_area = convex_w * convex_h + epsilon
        giou = iou - ((convex_area - union) / convex_area)
        return giou

    elif mode == 'ciou':
        convex_w = torch.max(box1_x2, box2_x2)-torch.min(box1_x1, box2_x1)
        convex_h = torch.max(box1_y2, box2_y2)-torch.min(box1_y1, box2_y1)
        c2 = convex_w ** 2 + convex_h ** 2 + epsilon  # convex diagonal squared
        rho2 = ((box2_x1 + box2_x2 - box1_x1 - box1_x2) ** 2 +
                (box2_y1 + box2_y2 - box1_y1 - box1_y2) ** 2) / 4  # center distance squared
        # some of the MsCoCo 2017 annotations before or after augmentations seem to have a box height of 0.0
        # this causes NaN values in loss due to dvision by zero in the ciou computation
        # for stability we add a small constant epsilon to box2_h and box1_1 above
        v = (4 / math.pi ** 2) * torch.pow(torch.atan(box2_w / box2_h) - torch.atan(box1_w / box1_h), 2)
        with torch.no_grad():
            alpha = v / ((1 + epsilon) - iou + v)
        ciou = iou - (rho2 / c2 + v * alpha)
        return ciou
    
    elif mode == 'siou':
        # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
        # Calculate center differences
        s_cw = (box2[..., 0:1] - box1[..., 0:1]) * 0.5 + epsilon
        s_ch = (box2[..., 1:2] - box1[..., 1:2]) * 0.5 + epsilon
        
        # Angle Cost
        sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
        sin_alpha = torch.abs(s_cw) / (sigma + epsilon)
        # sin_alpha = torch.where(sin_alpha > math.sin(math.pi/4), 
        #                     torch.abs(s_ch)/(sigma + epsilon), sin_alpha)  # Optional threshold
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
        
        # Distance Cost
        convex_w = torch.max(box1_x2, box2_x2) - torch.min(box1_x1, box2_x1)
        convex_h = torch.max(box1_y2, box2_y2) - torch.min(box1_y1, box2_y1)
        gamma = angle_cost - 2  # Different gamma calculation
        rho_x = (s_cw / (convex_w + epsilon)) ** 2
        rho_y = (s_ch / (convex_h + epsilon)) ** 2
        distance_cost = 2 - torch.exp(-gamma * rho_x) - torch.exp(-gamma * rho_y)
        
        # Shape Cost
        omega_w = torch.abs(box1_w - box2_w) / (torch.max(box1_w, box2_w) + epsilon)
        omega_h = torch.abs(box1_h - box2_h) / (torch.max(box1_h, box2_h) + epsilon)
        shape_cost = torch.pow(1 - torch.exp(-omega_w), 4) + torch.pow(1 - torch.exp(-omega_h), 4)
        
        # SIoU = IoU - (Distance Cost + Shape Cost) / 2
        siou = iou - (distance_cost + shape_cost) * 0.5 + epsilon
        return siou
    
    else:
        raise ValueError(f"Unknown mode: {mode}")

def box_intersection_over_union2(box1, box2, mode='iou'):
    """
    Calculates intersection of unions (IoU).
    Input: Boundbing box predictions (tensor) x, y, w, h (yolo format) of shape (N , 4)
            with N denoting the number of bounding boxes.
            Bounding box target/ground truth (tensor) x1, x2, y1, y2 of shape (N, 4).
            box format whether midpoint location or corner location of bounding boxes
            are used.
    Output: Intersection over union (tensor).
    """
    epsilon = 1e-9
    
    # Vectorized coordinate computation - compute all at once
    # Pred boxes
    box1_x1 = box1[..., 0:1] - (box1[..., 2:3] / 2)
    box1_y1 = box1[..., 1:2] - (box1[..., 3:4] / 2)
    box1_x2 = box1[..., 0:1] + (box1[..., 2:3] / 2)
    box1_y2 = box1[..., 1:2] + (box1[..., 3:4] / 2)
    
    # True boxes
    box2_x1 = box2[..., 0:1] - (box2[..., 2:3] / 2)
    box2_y1 = box2[..., 1:2] - (box2[..., 3:4] / 2)
    box2_x2 = box2[..., 0:1] + (box2[..., 2:3] / 2)
    box2_y2 = box2[..., 1:2] + (box2[..., 3:4] / 2)
    
    # Optimized area computation - avoid redundant operations
    box1_w = (box1_x2 - box1_x1).clamp(0)
    box1_h = (box1_y2 - box1_y1).clamp(0)
    box1_area = box1_w * box1_h + epsilon
    
    box2_w = (box2_x2 - box2_x1).clamp(0)
    box2_h = (box2_y2 - box2_y1).clamp(0)
    box2_area = box2_w * box2_h + epsilon
    
    # Intersection boxes
    inter_x1 = torch.max(box1_x1, box2_x1)
    inter_y1 = torch.max(box1_y1, box2_y1)
    inter_x2 = torch.min(box1_x2, box2_x2)
    inter_y2 = torch.min(box1_y2, box2_y2)
    
    inter_w = (inter_x2 - inter_x1).clamp(0)
    inter_h = (inter_y2 - inter_y1).clamp(0)
    inter_area = inter_w * inter_h + epsilon
    
    union = box1_area + box2_area - inter_area + epsilon

    # Compute IoU
    iou = inter_area / union

    if mode == 'iou':
        return iou

    elif mode == 'giou':
        # Convex diagonal - reuse computed values
        convex_w = torch.max(box1_x2, box2_x2) - torch.min(box1_x1, box2_x1)
        convex_h = torch.max(box1_y2, box2_y2) - torch.min(box1_y1, box2_y1)
        convex_area = convex_w * convex_h + epsilon
        giou = iou - ((convex_area - union) / convex_area)
        return giou

    elif mode == 'ciou':
        # Optimized CIoU - reuse values and reduce operations
        convex_w = torch.max(box1_x2, box2_x2) - torch.min(box1_x1, box2_x1)
        convex_h = torch.max(box1_y2, box2_y2) - torch.min(box1_y1, box2_y1)
        c2 = convex_w ** 2 + convex_h ** 2 + epsilon
        
        # Optimized center distance calculation
        center_x1 = (box1_x1 + box1_x2) / 2
        center_y1 = (box1_y1 + box1_y2) / 2
        center_x2 = (box2_x1 + box2_x2) / 2
        center_y2 = (box2_y1 + box2_y2) / 2
        rho2 = ((center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2)
        
        # Aspect ratio - use precomputed widths and heights
        v = (4 / math.pi ** 2) * torch.pow(torch.atan(box2_w / (box2_h + epsilon)) - 
                                          torch.atan(box1_w / (box1_h + epsilon)), 2)
        
        with torch.no_grad():
            alpha = v / (1 - iou + v + epsilon)
        
        ciou = iou - (rho2 / c2 + v * alpha)
        return ciou
    
    elif mode == 'siou':
        # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
        # Calculate center differences - optimized
        center_x1 = (box1_x1 + box1_x2) / 2
        center_y1 = (box1_y1 + box1_y2) / 2
        center_x2 = (box2_x1 + box2_x2) / 2
        center_y2 = (box2_y1 + box2_y2) / 2
        
        s_cw = (center_x2 - center_x1) * 0.5 + epsilon
        s_ch = (center_y2 - center_y1) * 0.5 + epsilon
        
        # Angle Cost
        sigma = torch.sqrt(s_cw ** 2 + s_ch ** 2 + epsilon)  # Use sqrt instead of pow(0.5)
        sin_alpha = torch.abs(s_cw) / (sigma + epsilon)
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
        
        # Distance Cost - reuse convex dimensions
        convex_w = torch.max(box1_x2, box2_x2) - torch.min(box1_x1, box2_x1)
        convex_h = torch.max(box1_y2, box2_y2) - torch.min(box1_y1, box2_y1)
        gamma = angle_cost - 2
        rho_x = (s_cw / (convex_w + epsilon)) ** 2
        rho_y = (s_ch / (convex_h + epsilon)) ** 2
        distance_cost = 2 - torch.exp(-gamma * rho_x) - torch.exp(-gamma * rho_y)
        
        # Shape Cost - use precomputed widths and heights
        omega_w = torch.abs(box1_w - box2_w) / (torch.max(box1_w, box2_w) + epsilon)
        omega_h = torch.abs(box1_h - box2_h) / (torch.max(box1_h, box2_h) + epsilon)
        shape_cost = torch.pow(1 - torch.exp(-omega_w), 4) + torch.pow(1 - torch.exp(-omega_h), 4)
        
        # SIoU = IoU - (Distance Cost + Shape Cost) / 2
        siou = iou - (distance_cost + shape_cost) * 0.5
        return siou
    
    else:
        raise ValueError(f"Unknown mode: {mode}")

def intersection_over_union(box1, box2):
    """Compute IoU between two bbox sets

    Arguments:
        box1 (tensor): tensor of shape (N, 4)
        box2 (tensor): tensor of shape (M, 4)

    Returns:
        tensor of shape (N, M) representing pair-by-pair iou values
        between two bbox sets.

    NOTES: box format (x1, y1, x2, y2)
    """
    epsilon = 1e-16
    N = box1.size(0)
    M = box2.size(0)
    # converts boxes
    x1 = box1[..., 0:1] - (box1[..., 2:3] / 2)
    y1 = box1[..., 1:2] - (box1[..., 3:4] / 2)
    x2 = box1[..., 0:1] + (box1[..., 2:3] / 2)
    y2 = box1[..., 1:2] + (box1[..., 3:5] / 2)
    box1 = torch.cat((x1, y1, x2, y2), dim=-1)

    x1 = box2[..., 0:1] - (box2[..., 2:3] / 2)
    y1 = box2[..., 1:2] - (box2[..., 3:4] / 2)
    x2 = box2[..., 0:1] + (box2[..., 2:3] / 2)
    y2 = box2[..., 1:2] + (box2[..., 3:5] / 2)
    box2 = torch.cat((x1, y1, x2, y2), dim=-1)

    # Compute intersection area
    lt = torch.max(
            box1[..., :2].unsqueeze(1).expand(N, M, 2), # (N, 2) -> (N, M, 2)
            box2[..., :2].unsqueeze(0).expand(N, M, 2), # (M, 2) -> (N, M, 2)
            )
    rb = torch.min(
            box1[..., 2:].unsqueeze(1).expand(N, M, 2), # (N, 2) -> (N, M, 2)
            box2[..., 2:].unsqueeze(0).expand(N, M, 2), # (M, 2) -> (N, M, 2)
            )
    wh = rb - lt                    # (N, M, 2)
    wh[wh<0] = 0                    # Non-overlapping conditions
    inter = wh[..., 0] * wh[..., 1] # (N, M)
    # Compute respective areas of boxes
    area1 = (box1[..., 2]-box1[..., 0]) * (box1[..., 3]-box1[..., 1]) # (N,)
    area2 = (box2[..., 2]-box2[..., 0]) * (box2[..., 3]-box2[..., 1]) # (M,)
    area1 = area1.unsqueeze(1).expand(N,M) # (N, M)
    area2 = area2.unsqueeze(0).expand(N,M) # (N, M)
    # Compute IoU
    iou = inter / (area1+area2-inter+epsilon)
    return iou.clamp(0)


def mean_average_precision(rank, pred_boxes, true_boxes, iou_threshold=0.5, nclasses=80):
    """
    Calculates mean average precision, by collecting predicted bounding boxes on the
    test set and then evaluate whether predictied boxes are TP or FP. Prediction with an
    IOU larger than 0.5 are TP and predictions larger than 0.5 are FP. Since there can be
    more than a single bounding box for an object, TP and FP are ordered by their confidence
    score or class probability in descending order, where the precision is computed as
    precision = (TP / (TP + FP)) and recall is computed as recall = (TP /(TP + FN)).

    Input: Predicted bounding boxes (list): [training index, class prediction C,
                                              probability score p, x1, y1, x2, y2], ,[...]
            Target/True bounding boxes:
    Output: Mean average precision (float)
    """
    eps = 1e-6
    avg_precision = []
    recall_lst = []
    precision_lst = []
    pred_boxes = torch.tensor(pred_boxes)
    true_boxes = torch.tensor(true_boxes)

    # if no predictions return 0
    if pred_boxes.size(0) == 0:
       return torch.tensor(0.0).to(rank), torch.tensor(0.0).to(rank), torch.tensor(0.0).to(rank)

    # iterate over class category c
    for c in range(nclasses):
        # init candidate detections and ground truth as an empty list for storage
        candidate_detections = []
        ground_truths = []

        # iterate over candidate bouding box predictions
        # index 1 is the class prediction and if equal to class c we are currently
        # looking at, then we append
        # if the candidate detection in the bounding box predictions is equal
        # to the class category c we are currently looking at add it to candidate list
        candidate_detections = pred_boxes[pred_boxes[..., 1] == c]
        ground_truths = true_boxes[true_boxes[..., 1] == c] 
        total_true_bboxes = ground_truths.size(0)
        total_pred_bboxes = candidate_detections.size(0)
        if total_true_bboxes == 0 or total_pred_bboxes == 0:
            continue
        # first index 0 is the training index, given image zero with 3 bbox
        # and img 1 has 5 bounding boxes, Counter will count how many bboxes
        # and create a dictionary, so amount_bbox = [0:3, 1:5]
        amount_bboxes = Counter([gt[0].item() for gt in ground_truths])
        # go through each key, val in dic
        # and convert: i.e., example below

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
        
        order = candidate_detections[..., 2].argsort(descending = True)
        candidate_detections = candidate_detections[order]
        ground_truths = ground_truths 
        # sort over probability scores

        # length for true positives and false positives for class based on detection
        # initalise tensors of zeros for true positives (TP) and false positives
        # (FP) as the length of possible candidate detections for a given class C
        TPs = torch.zeros((len(candidate_detections)))
        FPs = torch.zeros((len(candidate_detections)))

        for detection_idx in amount_bboxes.keys():
            offsets = torch.where(candidate_detections[..., 0] == detection_idx)[0]
            preds = candidate_detections[candidate_detections[..., 0] == detection_idx]
            # use only the ground_truths that have the same training idx as detection
            gts = ground_truths[ground_truths[..., 0] == detection_idx]

            # exception handling
            if preds.size(0) == 0 and gts.size(0) == 0:
                for offset in offsets:
                    FP[offset] = 1
            if preds.size(0) == 0 and gts.size(0) == 0:
                continue

            iou_mat = intersection_over_union(preds[..., 3:], gts[..., 3:])
            for pred_idx, ious in enumerate(iou_mat):
                best_idx = -1
                best_iou = 0
                for gt_idx, iou in enumerate(ious):
                    if (
                        iou > best_iou
                        and iou > iou_threshold
                        and amount_bboxes[detection_idx][gt_idx] == 0
                    ):
                        best_iou = iou
                        best_idx = gt_idx
                if best_idx != -1:
                    amount_bboxes[detection_idx][best_idx] = 1
                    TPs[offsets[pred_idx]] = 1
                else:
                    FPs[offsets[pred_idx]] = 1

        TP_cumsum = torch.cumsum(TPs, dim=0)
        FP_cumsum = torch.cumsum(FPs, dim=0)

        recalls = TP_cumsum / (total_true_bboxes + eps)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + eps)

        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        avg_precision.append(torch.trapz(precisions, recalls))

        TP = torch.sum(TPs)
        FP = torch.sum(FPs)
        recall = TP / (total_true_bboxes + eps)
        precision = TP / (TP + FP + eps)
        recall_lst.append(recall)
        precision_lst.append(precision)

    recall_val = sum(recall_lst) / len(recall_lst) if len(recall_lst) > 0 else torch.tensor(0.0)
    precision_val = sum(precision_lst) / len(precision_lst) if len(precision_lst) > 0 else torch.tensor(0.0)
    map_val = sum(avg_precision) / len(avg_precision) if len(avg_precision) > 0 else torch.tensor(0.0)

    # return map, recall and precision
    return map_val.to(rank), recall_val.to(rank), precision_val.to(rank)


def non_max_suppression_classwise(boxes, iou_threshold, confidence_threshold, max_vals_keep=1000):
    """
    Class-wise Non-Maximum Suppression (NMS) for YOLO bounding boxes.

    Arguments:
        boxes (tensor or list): shape (N, 6) in YOLO format (class, confidence, x, y, w, h)
        iou_threshold (float): IoU threshold for suppression
        confidence_threshold (float): minimum confidence to keep a box
        max_vals_keep (int): max number of boxes to process

    Returns:
        List of boxes after class-wise NMS
    """
    boxes = torch.tensor(boxes)
    if boxes.size(0) == 0:
        return []

    # Keep top max_vals_keep by confidence
    if boxes.size(0) > max_vals_keep:
        topk = boxes[:, 1].topk(max_vals_keep)
        boxes = boxes[topk.indices]

    # Filter by confidence
    boxes = boxes[boxes[:, 1] >= confidence_threshold]
    if boxes.size(0) == 0:
        return []

    final_boxes = []

    # Process each class separately
    classes = boxes[:, 0].unique()
    for cls in classes:
        cls_mask = boxes[:, 0] == cls
        cls_boxes = boxes[cls_mask]

        keep = []
        x1 = cls_boxes[:, 2] - cls_boxes[:, 4] / 2
        y1 = cls_boxes[:, 3] - cls_boxes[:, 5] / 2
        x2 = cls_boxes[:, 2] + cls_boxes[:, 4] / 2
        y2 = cls_boxes[:, 3] + cls_boxes[:, 5] / 2
        scores = cls_boxes[:, 1]
        areas = (x2 - x1) * (y2 - y1)

        order = scores.sort(descending=True)[1]

        while order.numel() > 0:
            i = order[0].item()
            keep.append(i)
            if order.numel() == 1:
                break

            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])

            inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            idx = (iou <= iou_threshold).nonzero(as_tuple=False).squeeze(1)
            order = order[idx + 1]

        final_boxes.extend(cls_boxes[keep].tolist())

    return final_boxes

def non_max_suppression(boxes, iou_threshold, confidence_threshold, max_vals_keep = 1000):
    """Non-maximal surpression on yolo bounding boxes using intersection over union (IoU)

    Arguments:
        boxes (tensor): tensor of shape (N, 6) in yolo format (class, confidence score, x, y, w, h)
        iou_threshold (float): minimum valid box confidence threshold

    Returns:
        boxes (tensor): tensor of shape (N, 6)

    """
    boxes = torch.tensor(boxes)
    if boxes.size(0) <= 1:
        return boxes
    
    if boxes.size(0) > max_vals_keep:
        topk = boxes[:, 1].topk(max_vals_keep)
        boxes = boxes[topk.indices]

    # create confidence mask to filter out bounding boxes that are below confidence threshold
    confidence_mask = boxes[:, 1] >= confidence_threshold
    boxes =  boxes[confidence_mask]
    # convert boxes
    x1 = boxes[..., 2:3] - (boxes[..., 4:5] / 2)
    y1 = boxes[..., 3:4] - (boxes[..., 5:6] / 2)
    x2 = boxes[..., 2:3] + (boxes[..., 4:5] / 2)
    y2 = boxes[..., 3:4] + (boxes[..., 5:6] / 2)
    scores = boxes[..., 1:2]
    # Area of shape (N,)
    areas = (x2-x1)*(y2-y1)

    keep = []
    order = scores.sort(0, descending=True)[1].squeeze(1)

    c = 0
    while order.numel() > 0:
        c = c + 1
        if order.numel() == 1:
            keep.append(order[0].item())
            break
        else:
            i = order[0].item()
            keep.append(i)
        # Compute IoU with remaining boxes (N-1,)
        xx1 = x1[order[1:]].clamp(min=x1[i])
        yy1 = y1[order[1:]].clamp(min=y1[i])
        xx2 = x2[order[1:]].clamp(max=x2[i])
        yy2 = y2[order[1:]].clamp(max=y2[i])
        inter = (xx2-xx1).clamp(min=0)*(yy2-yy1).clamp(min=0)
        iou = (inter / (areas[i]+areas[order[1:]]-inter)).squeeze(1)

        idx = (iou <= iou_threshold).nonzero().squeeze(1)

        if idx.numel() == 0:
            break
        order = order[idx+1]

    return  boxes[keep].tolist()

def top1accuracy(class_prob, target):
    """
    Calculates top 1 accuracy.
    Input: Class probabilities from the neural network (tensor)
    and target class predictions (tensor) of shape number of classes by batch size 
    Output: Top 1 accuracy (float).
    """
    with torch.no_grad():
        pred_class = torch.argmax(class_prob, dim = 1)
        top1_acc = sum(target==pred_class) / len(target)
    return top1_acc

def top5accuracy(class_prob, target):
    """
    Calculates top 1 accuracy.
    Input: Output of class probabilities from the neural network (tensor)
    of shape number of classes by batch size.
    Output: Top 5 accuracy (float).
    """
    with torch.no_grad():
        _, top5_class_pred = class_prob.topk(5, 1, largest = True, sorted = True)
        top5_class_pred = top5_class_pred.t()
        target_reshaped = target.view(1, -1).expand_as(top5_class_pred)
        correct = (top5_class_pred == target_reshaped)
        ncorrect_top5 = 0
        for i in range(correct.shape[1]):
            if (sum(correct[:,i]) >= 1):
                ncorrect_top5 = ncorrect_top5 + 1
        top5_acc = ncorrect_top5 / len(target)
        return top5_acc

def strip_square_brackets(pathtotxt):
    with open(pathtotxt, 'r') as my_file:
        text = my_file.read()
        text = text.replace("[","")
        text = text.replace("]","")
    with open(pathtotxt, 'w') as my_file:
        my_file.write(text)

def draw_bounding_box(image, bounding_boxes, test = False):
    """
    Input: PIL image and bounding boxes (as list).
    Output: Image with drawn bounding boxes.
    """
    image = np.ascontiguousarray(image, dtype = np.uint8)
    cmap = [
        [147, 69, 52],
        [29, 178, 255],
        [200, 149, 255],
        [151, 157, 255],
        [255, 115, 100],
        [134, 219, 61],
        [199, 55, 255],
        [49, 210, 207],
        [187, 212, 0],
        [52, 147, 26],
        [236, 24, 0],
        [168, 153, 44],
        [56, 56, 255],
        [255, 194, 0],
        [255, 56, 132],
        [133, 0, 82],
        [255, 56, 203],
        [31, 112, 255],
        [23, 204, 146]
        ]

    class_names = ['Person', 'Bicycle', 'Car', 'Motorcycle', 'Airplane', 'Bus', 'Train', 'Truck', 'Boat',
                'Traffic light', 'Fire hydrant', 'Stop sign', 'Parking meter', 'Bench', 'Bird', 'Cat',
                'Dog', 'Horse', 'Sheep', 'Cow', 'Elephant', 'Bear', 'Zebra', 'Giraffe', 'Backpack',
                'Umbrella', 'Handbag', 'Tie', 'Suitcase', 'Frisbee', 'Skis', 'Snowboard', 'Sports ball',
                'Kite', 'Baseball bat', 'Baseball glove', 'Skateboard', 'Surfboard', 'Tennis racket',
                'Bottle', 'Wine glass', 'Cup', 'Fork', 'Knife', 'Spoon', 'Bowl', 'Banana', 'Apple',
                'Sandwich', 'Orange', 'Broccoli', 'Carrot', 'Hot dog', 'Pizza', 'Donut', 'Cake', 'Chair',
                'Couch', 'Potted plant', 'Bed', 'Dining table', 'Toilet', 'Tv', 'Laptop', 'Mouse', 'Remote',
                'Keyboard', 'Cell phone', 'Microwave', 'Oven', 'Toaster', 'Sink', 'Refrigerator', 'Book',
                'Clock', 'Vase', 'Scissors', 'Teddy bear', 'Hair drier', 'Toothbrush'
                ]

    # Calculate the number of classes
    nclasses = len(class_names)

    # Create a list of colors by mapping the class index to the cmap list
    colors = [cmap[i % len(cmap)] for i in range(nclasses)]

    # Extract transform_vals
    for i in range(len(bounding_boxes)):
        if test:
            height, width = image.shape[:2]
            class_pred = int(bounding_boxes[i][0])
            certainty = bounding_boxes[i][1]
            bounding_box = bounding_boxes[i][2:]

            # Note: width and heigh indexes are switches, somewhere, these are switched so
            # we correct for the switch by switching
            #bounding_box[2], bounding_box[3] = bounding_box[3], bounding_box[2]
            #assert len(bounding_box) == 4, "Bounding box prediction exceed x, y ,w, h."
            # Extract x, midpoint, y midpoint, w width and h height
            x = bounding_box[0]
            y = bounding_box[1]
            w = bounding_box[2]
            h = bounding_box[3]

        else:
            height, width = image.shape[:2]
            class_pred = int(bounding_boxes[i][0])
            bounding_box = bounding_boxes[i][2:]

            #assert len(bounding_box) == 4, "Bounding box prediction exceed x, y ,w, h."
            # Extract x midpoint, y midpoint, w width and h height
            x = bounding_box[0]
            y = bounding_box[1]
            w = bounding_box[2]
            h = bounding_box[3]

        l = int((x - w / 2) * width)
        r = int((x + w / 2) * width)
        t = int((y - h / 2) * height)
        b = int((y + h / 2) * height)

        if l < 0:
            l = 0
        if r > width - 1:
            r = width - 1
        if t < 0:
            t = 0
        if b > height - 1:
            b = height - 1

        image = cv.rectangle(image, (l, t), (int(r), int(b)), colors[class_pred], 3)
        (txt_width, txt_height), _ = cv.getTextSize(class_names[class_pred], cv.FONT_HERSHEY_TRIPLEX, 0.6, 2)

        if t < 20:
            image = cv.rectangle(image, (l-2, t + 15), (l + txt_width, t), colors[class_pred], -1)
            image = cv.putText(image, class_names[class_pred], (l, t+12),
                    cv.FONT_HERSHEY_TRIPLEX, 0.5, [255, 255, 255], 1)
        else:
            image = cv.rectangle(image, (l-2, t - 15), (l + txt_width, t), colors[class_pred], -1)
            image = cv.putText(image, class_names[class_pred], (l, t-3),
                    cv.FONT_HERSHEY_TRIPLEX, 0.5, [255, 255, 255], 1)

    return image

def draw_bounding_box_vid(image, bounding_boxes, test = False):
    """
    Input: PIL image and bounding boxes (as list).
    Output: Image with drawn bounding boxes.
    """
    image = np.ascontiguousarray(image, dtype = np.uint8)
    cmap = [
        [147, 69, 52],
        [29, 178, 255],
        [200, 149, 255],
        [151, 157, 255],
        [255, 115, 100],
        [134, 219, 61],
        [199, 55, 255],
        [49, 210, 207],
        [187, 212, 0],
        [52, 147, 26],
        [236, 24, 0],
        [168, 153, 44],
        [56, 56, 255],
        [255, 194, 0],
        [255, 56, 132],
        [133, 0, 82],
        [255, 56, 203],
        [31, 112, 255],
        [23, 204, 146]
        ]

    class_names = ["Airplane", "Antelope", "Bear", "Bicycle", "Bird", "Bus", "Car",
                     "Cattle", "Dog", "Domestic cat", "Elephant", "Fox", "Giant panda",
                     "Hamster", "Horse", "Lion", "Lizard", "Monkey", "Motorcycle", "Rabbit",
                     "Red panda", "Sheep", "Snake", "Squirrel", "Tiger", "Train",
                     "Turtle", "Watercraft", "Whale", "Zebra"
                     ]


    # Calculate the number of classes
    nclasses = len(class_names)

    # Create a list of colors by mapping the class index to the cmap list
    colors = [cmap[i % len(cmap)] for i in range(nclasses)]

    # Extract transform_vals
    for i in range(len(bounding_boxes)):
        if test:
            height, width = image.shape[:2]
            class_pred = int(bounding_boxes[i][0])
            certainty = bounding_boxes[i][1]
            bounding_box = bounding_boxes[i][2:]

            # Note: width and heigh indexes are switches, somewhere, these are switched so
            # we correct for the switch by switching
            #bounding_box[2], bounding_box[3] = bounding_box[3], bounding_box[2]
            #assert len(bounding_box) == 4, "Bounding box prediction exceed x, y ,w, h."
            # Extract x, midpoint, y midpoint, w width and h height
            x = bounding_box[0]
            y = bounding_box[1]
            w = bounding_box[2]
            h = bounding_box[3]

        else:
            height, width = image.shape[:2]
            class_pred = int(bounding_boxes[i][0])
            bounding_box = bounding_boxes[i][2:]

            #assert len(bounding_box) == 4, "Bounding box prediction exceed x, y ,w, h."
            # Extract x midpoint, y midpoint, w width and h height
            x = bounding_box[0]
            y = bounding_box[1]
            w = bounding_box[2]
            h = bounding_box[3]

        l = int((x - w / 2) * width)
        r = int((x + w / 2) * width)
        t = int((y - h / 2) * height)
        b = int((y + h / 2) * height)

        if l < 0:
            l = 0
        if r > width - 1:
            r = width - 1
        if t < 0:
            t = 0
        if b > height - 1:
            b = height - 1

        image = cv.rectangle(image, (l, t), (int(r), int(b)), colors[class_pred], 3)
        (txt_width, txt_height), _ = cv.getTextSize(class_names[class_pred], cv.FONT_HERSHEY_TRIPLEX, 0.6, 2)

        if t < 20:
            image = cv.rectangle(image, (l-2, t + 15), (l + txt_width, t), colors[class_pred], -1)
            image = cv.putText(image, class_names[class_pred], (l, t+12),
                    cv.FONT_HERSHEY_TRIPLEX, 0.5, [255, 255, 255], 1)
        else:
            image = cv.rectangle(image, (l-2, t - 15), (l + txt_width, t), colors[class_pred], -1)
            image = cv.putText(image, class_names[class_pred], (l, t-3),
                    cv.FONT_HERSHEY_TRIPLEX, 0.5, [255, 255, 255], 1)

    return image

def class_accuracy(rank, model_pred, target, confidence_threshold):
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0
    with torch.no_grad():
        for i in range(3):
            target[i] = target[i].to(rank)
            obj = target[i][..., 0] == 1 # in paper this is Iobj_i
            noobj = target[i][..., 0] == 0  # in paper this is Iobj_i

            correct_class += torch.sum(torch.argmax(model_pred[i][..., 5:][obj], dim=-1) == target[i][..., 5][obj])
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(model_pred[i][..., 0]) > confidence_threshold
            correct_obj += torch.sum(obj_preds[obj] == target[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj] == target[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)

        class_acc = (correct_class / (tot_class_preds + 1e-16))
        noobj_acc = (correct_noobj / (tot_noobj + 1e-16))
        obj_acc = (correct_obj / (tot_obj + 1e-16))
    return class_acc, noobj_acc, obj_acc

def class_accuracy_dio(rank, model_pred, target, confidence_threshold):
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0
    with torch.no_grad():
        for i in range(2):
            target[i] = target[i].to(rank)
            obj = target[i][..., 0] == 1 # in paper this is Iobj_i
            noobj = target[i][..., 0] == 0  # in paper this is Iobj_i

            correct_class += torch.sum(torch.argmax(model_pred[i][..., 5:][obj], dim=-1) == target[i][..., 5][obj])
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(model_pred[i][..., 0]) > confidence_threshold
            correct_obj += torch.sum(obj_preds[obj] == target[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj] == target[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)

        class_acc = (correct_class / (tot_class_preds + 1e-16))
        noobj_acc = (correct_noobj / (tot_noobj + 1e-16))
        obj_acc = (correct_obj / (tot_obj + 1e-16))
    return class_acc, noobj_acc, obj_acc

def class_accuracy_enas(rank, model_pred, target, confidence_threshold):
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0
    with torch.no_grad():
        for i in range(1):
            target[i] = target[i].to(rank)
            obj = target[i][..., 0] == 1 # in paper this is Iobj_i
            noobj = target[i][..., 0] == 0  # in paper this is Iobj_i

            correct_class += torch.sum(torch.argmax(model_pred[..., 5:][obj], dim=-1) == target[i][..., 5][obj])
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(model_pred[..., 0]) > confidence_threshold
            correct_obj += torch.sum(obj_preds[obj] == target[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj] == target[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)

        class_acc = (correct_class / (tot_class_preds + 1e-16))
        noobj_acc = (correct_noobj / (tot_noobj + 1e-16))
        obj_acc = (correct_obj / (tot_obj + 1e-16))
    return class_acc, noobj_acc, obj_acc

def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    """
    Scales the predictions coming from the model relative to the image.

    Args:
        predictions (Tensor): A tensor of size (N, 3, S, S, nclasses+5), containing model predictions.
        anchors (list of tuples): The anchors used for the predictions.
        S (int): The number of cells the image is divided into on the width (and height).
        is_preds (bool, optional): Whether the input is predictions or the true bounding boxes. Default is True.

    Returns:
        list of lists: The converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index,
                      object score, bounding box coordinates.
    """
    with torch.no_grad():
        batch_size = predictions.shape[0]
        num_anchors = len(anchors)
        box_predictions = predictions[..., 1:5]
        if is_preds:
            anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
            box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
            box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
            scores = torch.sigmoid(predictions[..., 0:1])
            best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
        else:
            scores = predictions[..., 0:1]
            best_class = predictions[..., 5:6]

        cell_indices = (
            torch.arange(S)
            .repeat(predictions.shape[0], 3, S, 1)
            .unsqueeze(-1)
            .to(predictions.device)
        )

        x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
        y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
        w_h = 1 / S * box_predictions[..., 2:4]
        converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(batch_size, num_anchors * S * S, 6)
        
    return converted_bboxes.tolist()


def get_bounding_boxes(rank, loader, model, iou_threshold, anchors, confidence_threshold):
    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, (x, y) in enumerate(loader):
        x = x.permute(0, 3, 1, 2)
        x = x.to(torch.float32).to(rank)
        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        for i in range(3):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(rank) * S
            boxes_scale_i = cells_to_bboxes(
                predictions[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box
        # we just want one bbox for each label, not one for each scale
        true_bboxes = cells_to_bboxes(
            y[2], anchor, S=S, is_preds=False
        )
        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                confidence_threshold=confidence_threshold)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)
            for box in true_bboxes[idx]:
                if box[1] > confidence_threshold:
                    all_true_boxes.append([train_idx] + box)
            train_idx += 1
    model.train()
    return all_pred_boxes, all_true_boxes

def get_bounding_boxes_vid(rank, loader, model, iou_threshold, anchors, confidence_threshold):
    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, (x, y) in enumerate(loader):
        for seq_idx in range(len(x)):
            x_t = x[seq_idx].permute(0, 3, 1, 2)
            x_t = x_t.to(torch.float32).to(rank)
            with torch.no_grad():
                predictions = model(x_t)
            batch_size = x_t.shape[0]
            bboxes = [[] for _ in range(batch_size)]
            for i in range(3):
                S = predictions[i].shape[2]
                anchor = torch.tensor([*anchors[i]]).to(rank) * S
                boxes_scale_i = cells_to_bboxes(
                    predictions[i], anchor, S=S, is_preds=True
                )
                for idx, (box) in enumerate(boxes_scale_i):
                    bboxes[idx] += box
            # we just want one bbox for each label, not one for each scale
            true_bboxes = cells_to_bboxes(
                y[seq_idx][2], anchor, S=S, is_preds=False
            )
            for idx in range(batch_size):
                nms_boxes = non_max_suppression(
                    bboxes[idx],
                    iou_threshold=iou_threshold,
                    confidence_threshold=confidence_threshold)

                for nms_box in nms_boxes:
                    all_pred_boxes.append([train_idx] + nms_box)
                for box in true_bboxes[idx]:
                    if box[1] > confidence_threshold:
                        all_true_boxes.append([train_idx] + box)
                train_idx += 1
    model.train()
    return all_pred_boxes, all_true_boxes


def get_bounding_boxes_holo_enas_vid(rank, loader, model, iou_threshold, anchors, confidence_threshold):
    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, (x, y) in enumerate(loader):
        carry = ((None, None), (None, None), (None, None), (None, None))
        last_ts = ((0), (0), (0), (0))
        for seq_idx in range(len(x)):
            x_t = x[seq_idx].permute(0, 3, 1, 2)
            x_t = x_t.to(torch.float32).to(rank)
            with torch.no_grad():
                predictions, carry = model(x_t, t=seq_idx, carry=carry)
                # add temoral metric here

            batch_size = x_t.shape[0]
            bboxes = [[] for _ in range(batch_size)]
            for i in range(1):
                S = predictions.shape[2]       
                anchor = torch.tensor([*anchors[i]]).to(rank) * S

                boxes_scale_i = cells_to_bboxes(
                    predictions, anchor, S=S, is_preds=True
                )
                for idx, (box) in enumerate(boxes_scale_i):
                    bboxes[idx] += box
            # we just want one bbox for each label, not one for each scale
            true_bboxes = cells_to_bboxes(
                y[seq_idx][0], anchor, S=S, is_preds=False
            )
            for idx in range(batch_size):
                nms_boxes = non_max_suppression(
                    bboxes[idx],
                    iou_threshold=iou_threshold,
                    confidence_threshold=confidence_threshold)

                for nms_box in nms_boxes:
                    all_pred_boxes.append([train_idx] + nms_box)
                for box in true_bboxes[idx]:
                    if box[1] > confidence_threshold:
                        all_true_boxes.append([train_idx] + box)
                train_idx += 1
    model.train()
    return all_pred_boxes, all_true_boxes

def get_bounding_boxes_holo_enas_vid_seq_wise(rank, loader, model, iou_threshold, anchors, confidence_threshold):
    # Make sure the model is in evaluation mode
    model.eval()
    all_pred_boxes_t = []    # List to store predictions for timestep t
    all_pred_boxes_t1 = []   # List to store predictions for timestep t+1

    # Loop through each batch of the data loader
    for batch_idx, (x, y) in enumerate(loader):
        carry = ((None, None), (None, None), (None, None), (None, None))

        # Loop through the sequence of timesteps and only consider t and t+1
        for seq_idx in range(len(x) - 1):  # We subtract 1 to ensure we have a t+1
            # Get predictions for timestep t
            x_t = x[seq_idx].permute(0, 3, 1, 2).to(torch.float32).to(rank)
            with torch.no_grad():
                predictions_t, carry = model(x_t, t=seq_idx, carry=carry)

            # Get predictions for timestep t+1
            x_t1 = x[seq_idx + 1].permute(0, 3, 1, 2).to(torch.float32).to(rank)
            with torch.no_grad():
                predictions_t1, carry = model(x_t1, t=seq_idx + 1, carry=carry)

            batch_size = x_t.shape[0]
            bboxes_t = [[] for _ in range(batch_size)]
            bboxes_t1 = [[] for _ in range(batch_size)]

            # Loop through the anchor scale (assuming one scale in your setup)
            for i in range(1):
                S = predictions_t.shape[2]
                anchor = torch.tensor([*anchors[i]]).to(rank) * S

                # Convert cell predictions to bounding boxes for timestep t
                boxes_scale_t = cells_to_bboxes(predictions_t, anchor, S=S, is_preds=True)
                for idx, box in enumerate(boxes_scale_t):
                    bboxes_t[idx] += box

                # Convert cell predictions to bounding boxes for timestep t+1
                boxes_scale_t1 = cells_to_bboxes(predictions_t1, anchor, S=S, is_preds=True)
                for idx, box in enumerate(boxes_scale_t1):
                    bboxes_t1[idx] += box

            # Apply NMS for predictions at timestep t and t+1 separately
            for idx in range(batch_size):
                # NMS for timestep t
                nms_boxes_t = non_max_suppression(
                    bboxes_t[idx],
                    iou_threshold=iou_threshold,
                    confidence_threshold=confidence_threshold
                )

                # NMS for timestep t+1
                nms_boxes_t1 = non_max_suppression(
                    bboxes_t1[idx],
                    iou_threshold=iou_threshold,
                    confidence_threshold=confidence_threshold
                )

                # Store the NMS results for each timestep separately
                all_pred_boxes_t.extend([[batch_idx, seq_idx] + box for box in nms_boxes_t])
                all_pred_boxes_t1.extend([[batch_idx, seq_idx + 1] + box for box in nms_boxes_t1])

    # Set the model back to training mode
    model.train()

    return all_pred_boxes_t, all_pred_boxes_t1

def get_bounding_boxes_yolo_enas_vid(rank, loader, model_yolo, model_rnn, iou_threshold, anchors, confidence_threshold):
    # make sure model is in eval before get bboxes
    model_yolo.eval()
    model_rnn.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, (x, y) in enumerate(loader):
        for seq_idx in range(len(x)):
            x_t = x[seq_idx].permute(0, 3, 1, 2)
            x_t = x_t.to(torch.float32).to(rank)
            with torch.no_grad():
                predictions = model_yolo(x_t)
                predictions = model_rnn(x_t)

            batch_size = x_t.shape[0]
            bboxes = [[] for _ in range(batch_size)]
            for i in range(1):
                S = predictions.shape[2]       
                anchor = torch.tensor([*anchors[i]]).to(rank) * S

                boxes_scale_i = cells_to_bboxes(
                    predictions, anchor, S=S, is_preds=True
                )
                for idx, (box) in enumerate(boxes_scale_i):
                    bboxes[idx] += box
            # we just want one bbox for each label, not one for each scale
 
            true_bboxes = cells_to_bboxes(
                y[seq_idx][0], anchor, S=S, is_preds=False
            )
            for idx in range(batch_size):
                nms_boxes = non_max_suppression(
                    bboxes[idx],
                    iou_threshold=iou_threshold,
                    confidence_threshold=confidence_threshold)

                for nms_box in nms_boxes:
                    all_pred_boxes.append([train_idx] + nms_box)
                for box in true_bboxes[idx]:
                    if box[1] > confidence_threshold:
                        all_true_boxes.append([train_idx] + box)
                train_idx += 1
    model_yolo.train()
    model_rnn.train()

    return all_pred_boxes, all_true_boxes


def get_bounding_boxes_holo_vid(rank, loader, model, iou_threshold, anchors, confidence_threshold):
    # Make sure model is in eval mode before getting bboxes
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            carry = ((None, None), (None, None), (None, None))
            for seq_idx in range(len(x)):
                x_t = x[seq_idx].permute(0, 3, 1, 2)
                x_t = x_t.to(torch.float32).to(rank)
                predictions, carry = model(x_t, t=seq_idx, carry=carry)
                batch_size = x_t.shape[0]
                bboxes = [[] for _ in range(batch_size)]
                for i in range(3):
                    S = predictions[i].shape[2]
                    anchor = torch.tensor([*anchors[i]]).to(rank) * S
                    boxes_scale_i = cells_to_bboxes(
                        predictions[i], anchor, S=S, is_preds=True
                    )
                    for idx, box in enumerate(boxes_scale_i):
                        bboxes[idx] += box
                # We just want one bbox for each label, not one for each scale
                true_bboxes = cells_to_bboxes(
                    y[seq_idx][2], anchor, S=S, is_preds=False
                )
                for idx in range(batch_size):
                    nms_boxes = non_max_suppression(
                        bboxes[idx],
                        iou_threshold=iou_threshold,
                        confidence_threshold=confidence_threshold
                    )

                    for nms_box in nms_boxes:
                        all_pred_boxes.append([train_idx] + nms_box)
                    for box in true_bboxes[idx]:
                        if box[1] > confidence_threshold:
                            all_true_boxes.append([train_idx] + box)
                    train_idx += 1
    
    model.train()
    return all_pred_boxes, all_true_boxes


def linearly_increasing_lr(initial_lr, final_lr, current_epoch, total_epochs):
    return initial_lr + (final_lr - initial_lr) * (current_epoch / total_epochs)

def count_parameters(model: nn.Module, adjust_complex: bool = True) -> int:
    def get_count(param: nn.Parameter) -> int:
        return param.numel() * (param.is_complex() + adjust_complex)

    return sum(get_count(p) for p in model.parameters())