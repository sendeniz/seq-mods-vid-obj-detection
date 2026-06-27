import torch
import numpy as np
from collections import Counter
import cv2 as cv
import torch.nn.functional as F 
from utils.dataset import CoCoDataset
#from utils.utils import get_bouding_boxes, mean_average_precision
from torch.utils.data import DataLoader
from models.yolov4 import YoloV4_EfficentNet
import torch.optim as optim
device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')

scales = [13, 26, 52]
anchors = [
       [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
       [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
       [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
       ]


conf_thresh = 0.5
nms_iou_thresh = 0.5
map_iou_thresh = 0.5
batch_size = 2
nworkers = 2
nclasses = 80
lr =  0.00001
weight_decay = 0.0005
path_cpt_file = f'cpts/yolov4_416_mscoco.cpt'
checkpoint = torch.load(path_cpt_file)
model = YoloV4_EfficentNet(nclasses = nclasses).to(device)
optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#model.eval()
print("Petrained YoloV4 416 EfficentNet S Net initalized.")

test_dataset = test_dataset = CoCoDataset("data/coco/test_2examples.csv", "data/coco/images/", "data/coco/labels/",
                          S = scales, anchors = anchors, mode = 'test')

test_loader = DataLoader(dataset = test_dataset, num_workers = nworkers,
                                            batch_size = batch_size,
                                            shuffle = False, drop_last = False)


def xywh2tlbr(boxes):
    """Convert bounding box format from 'xywh' to 'tlbr'

    Arguements:
        boxes (tensor): tensor of shape (N, 4)

    NOTE: box format is (x, y, w, h)
    """
    #print("xywh shape:", boxes.shape)
    #print("xywh:", boxes)
    x1 = boxes[..., 0:1]-(boxes[..., 2:3]/2)    # (N,1)
    y1 = boxes[..., 1:2]-(boxes[..., 3:4]/2)    # (N,1)
    x2 = boxes[..., 0:1]+(boxes[..., 2:3]/2)    # (N,1)
    y2 = boxes[..., 1:2]+(boxes[..., 3:4]/2)    # (N,1)
    tlbr = torch.cat([x1, y1, x2, y2], dim=-1)
    #print("xywh after:", tlbr)
    #print("tlbr:", tlbr)
    return tlbr

def cells_to_boxes(cells, scale):
    """Transform the coordinate system of prediction to image coordiante system

    Arguments:
        cells (tensor): tensor of shape (N, 3, scale, scale, 6(7))
        scale (int): the scale of image

    Returns:
        tensor of shape (N, 3, scale, scale, 6(7))
        the format of output is (x, y, w, h, conf, class, [mask])

    NOTES: the cells format is (x_offset, y_offset, w_cell, h_cell, conf, class)
    """
    N = cells.size(0)
    # Extract each dimension
    x_cells = cells[..., 0:1]   # (N, 3, scale, scale, 1)
    y_cells = cells[..., 1:2]   # (N, 3, scale, scale, 1)
    w_cells = cells[..., 2:3]   # (N, 3, scale, scale, 1)
    h_cells = cells[..., 3:4]   # (N, 3, scale, scale, 1)
    conf = cells[..., 4:5]      # (N, 3, scale, scale, 1)
    cls = cells[..., 5:6]       # (N, 3, scale, scale, 1)
    if cells.size(4) > 6:
        tails = cells[..., 6:]  # (N, 3, scale, scale, N)
    # Cell coordinates
    cell_indices = (            # (N, 3, scale, scale, 1)
        torch.arange(scale)
        .repeat(N, 3, scale, 1)
        .unsqueeze(-1)
        .to(cells.device)
        )
    # Convert coordinates
    x = (1/scale)*(x_cells+cell_indices)
    y = (1/scale)*(y_cells+cell_indices.permute(0, 1, 3, 2, 4))
    w = (1/scale)*(w_cells)
    h = (1/scale)*(h_cells)
    if cells.size(4) > 6:
        boxes = torch.cat([x, y, w, h, conf, cls, tails], dim=-1)
    else:
        #boxes = torch.cat([x, y, w, h, conf, cls], dim=-1)
        boxes = torch.cat([cls, conf, x, y, w, h], dim = -1)
    return boxes

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

def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5, n_classes=80):
    """Compute average precision related to all classes

    Arguments:
        pred_boxes (tensor): tensor of shape (N, 7)
        true_boxes (tensor): tensor of shape (M, 7)
        iou_threshold (float): true positive criterion threshold
        n_classes (int): number of classes

    Returns:
        a dictionary of key "mAP", "recall", and "precision"

    NOTE: The format of boxes is (idx, x1, y1, x2, y2, conf, class)
    """
    epsilon = 1e-6
    all_recalls = []
    all_precisions = []
    average_precisions = [] # Save a list of average precision of each class
    # Caculate average precision class-by-class
    print("###########################")
    print("pred_boxes:", pred_boxes)
    print("true_boxes:", true_boxes)
    print("pred_boxes shape:", pred_boxes.shape)
    print("true_boxes shape:", true_boxes.shape)
    print("pred_boxes type:", type(pred_boxes))
    print("true_boxes type:", type(true_boxes))
    for c in range(n_classes):
        # Filter out boxes of class 'c'
        detections = pred_boxes[pred_boxes[..., 0] == c]
        ground_truths = true_boxes[true_boxes[..., 0] == c]
        print("detections shape:", detections.shape)
        print("ground truths shape:", ground_truths.shape)
        # Exception handling
        print("ground truth size:", ground_truths.size())
        print("type ground truth:", type(ground_truths))
        print("ground_thruts:", ground_truths)
        total_true_bboxes = ground_truths.size(0)
        if total_true_bboxes == 0 or detections.size(0) == 0:
            #print("exception handling triggered")
            continue
        # print("exception not triggered")
        # Lookup table
        amount_bboxes = Counter([ gt[0].item() for gt in ground_truths ])
        for sample_idx, count in amount_bboxes.items():
            amount_bboxes[sample_idx] = torch.zeros(count)
        # Placeholder to keep information where a pred box is TP/FP
        TPs = torch.zeros(detections.size(0))
        FPs = torch.zeros(detections.size(0))

        # Descending detections by confidence score
        order = detections[..., 5].argsort(descending=True)
        detections = detections[order]

        for sample_idx in amount_bboxes.keys():
            offsets = torch.where(detections[..., 0] == sample_idx)[0]
            preds = detections[detections[..., 0] == sample_idx]
            gts = ground_truths[ground_truths[..., 0] == sample_idx]

            # Exception Handling
            if preds.size(0) != 0 and gts.size(0) == 0:
                for offset in offsets:
                    FP[offset] = 1 
            elif preds.size(0) == 0 or gts.size(0) == 0:
                continue

            iou_mat = intersection_over_union(preds[:, 2:], gts[:, 2:5])
            print("iou_mat:", iou_mat)
            for pred_idx, ious in enumerate(iou_mat):
                best_idx = -1
                best_iou = 0
                for gt_idx, iou in enumerate(ious):
                    if (
                        iou > best_iou
                        and iou > iou_threshold
                        and amount_bboxes[sample_idx][gt_idx] == 0
                    ):
                        best_iou = iou
                        best_idx = gt_idx
                if best_idx != -1:
                    amount_bboxes[sample_idx][best_idx] = 1
                    TPs[offsets[pred_idx]] = 1
                else:
                    FPs[offsets[pred_idx]] = 1

        TP_cumsum = torch.cumsum(TPs, dim=0)
        FP_cumsum = torch.cumsum(FPs, dim=0)
        recalls = TP_cumsum / (total_true_bboxes)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        average_precisions.append(torch.trapz(precisions, recalls))

        TP = torch.sum(TPs)
        FP = torch.sum(FPs)
        recall = TP / (total_true_bboxes+epsilon)
        precision = TP / (TP + FP + epsilon)
        all_recalls.append(recall)
        all_precisions.append(precision)
    #print("recall val:", recall)
    #print("precision val:", precision)
    #print("recall sum:", sum(all_recalls))
    #print("recall len:", len(all_recalls))
    recall = sum(all_recalls)/len(all_recalls)
    precision = sum(all_precisions)/len(all_precisions)
    mAP = sum(average_precisions) / len(average_precisions)
    return { "mAP": mAP, "recall": recall, "precision": precision }

def nms(boxes, scores, iou_threshold, extras=None):
    """Non-maximum suppression on bbox set

    Arguments:
        boxes (tensor): tensor of shape (N, 4)
        scores (tensor): tensor of shape (N, 1)
        iou_threshold (float): minimum valid box confidence threshold

    Returns:
        a tuple of tensors (boxes, scores) after doing nms filtering

    NOTES:
        box format (x1, y1, x2, y2)
        score is (objectness*class probability) => P(Class|Obj)
    """
    if boxes.size(0) <= 1:
        return boxes, scores, extras
    # Coordinate of shape (N,)
    x1 = boxes[..., 0]
    y1 = boxes[..., 1]
    x2 = boxes[..., 2]
    y2 = boxes[..., 3]
    # Area of shape (N,)
    areas = (x2-x1)*(y2-y1)

    keep = []
    order = scores.sort(0, descending=True)[1].squeeze(1)
    while order.numel() > 0:
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
        iou = (inter / (areas[i]+areas[order[1:]]-inter))
        idx = (iou <= iou_threshold).nonzero().squeeze(1)
        if idx.numel() == 0:
            break
        order = order[idx+1]

    return boxes[keep], scores[keep], (extras[keep] if extras is not None else None)

def nms_by_class(bboxes, target, iou_threshold):
    if bboxes.size(0) == 0:
        return []
    # Split the fields
    #print("nms by class bboxes:", bboxes)
    boxes = xywh2tlbr(bboxes[..., :4])
    scores = bboxes[..., 4:5]
    classes = bboxes[..., 5]
    extras = bboxes[..., 6:] if bboxes.size(1) > 6 else None
    # Filter out target class
    mask = (classes == target)
    if torch.sum(mask) == 0:
        return []
    # Perform nms on objects
    boxes, scores, extras = nms(boxes=boxes[mask],
                                scores=scores[mask],
                                iou_threshold=iou_threshold,
                                extras=extras[mask] if extras is not None else None)
    # Merge fields back
    classes = torch.tensor([[target]]).repeat(boxes.size(0), 1)
    columns = [ boxes, scores, classes ] + ([extras] if extras is not None else [])
    bboxes = torch.cat(columns, dim=1)
    #print("nms by class bboxes:", bboxes)
    return bboxes.tolist()

def check_map(model, test_loader, anchors, scales, conf_threshold = 0.5, nms_iou_threshold = 0.5, n_classes = 80):
    sample_idx = 0
    all_pred_bboxes = []
    all_true_bboxes = []
    model.eval()
    torch_anchors = torch.tensor(anchors) # (3, 3, 2)
    torch_scales = torch.tensor(scales) # (3,)
    scaled_anchors = (  # (3, 3, 2)
                torch_anchors * (
                    torch_scales
                    .unsqueeze(1)
                    .unsqueeze(1)
                    .repeat(1, 3, 2)
                    )
                )
    scaled_anchors = scaled_anchors.to(device)
    loop = tqdm(self.valid_loader, leave=True, desc="Check mAP")
    for batch_idx, (imgs, targets) in enumerate(loop):
        #print("targets:", targets)
        batch_size = imgs.size(0)
        # Move device
        imgs = imgs.to(device)             # (N, 3, 416, 416)
        target_s1 = targets[0].to(device)  # (N, 3, 13, 13, 6)
        target_s2 = targets[1].to(device)  # (N, 3, 26, 26, 6)
        target_s3 = targets[2].to(device)  # (N, 3, 52, 52, 6)
        targets = [ target_s1, target_s2, target_s3 ]
        # Model Forward
        imgs = imgs.permute(0, 3, 1, 2)
        with torch.no_grad():
            preds = model(imgs)
            # Convert cells to bboxes
            # =================================================================
            true_bboxes = [ [] for _ in range(batch_size) ]
            pred_bboxes = [ [] for _ in range(batch_size) ]
            print("Check map Predicted boxes:,", pred_bboxes)
            #print("Predicted boxes shape:", pred_bboxes.shape)
            print("Check map True boxes:", true_bboxes)
            #print("True bboxes shape:", true_bboxes.shape)
            for scale_idx, (pred, target) in enumerate(zip(preds, targets)):
                scale = pred.size(2) 
                anchors = scaled_anchors[scale_idx] # (3, 2)
                anchors = anchors.reshape(1, 3, 1, 1, 2) # (1, 3, 1, 1, 2)
                # Convert prediction to correct format
                pred[..., 0:2] = torch.sigmoid(pred[..., 0:2])      # (N, 3, S, S, 2)
                pred[..., 2:4] = torch.exp(pred[..., 2:4])*anchors  # (N, 3, S, S, 2)
                pred[..., 4:5] = torch.sigmoid(pred[..., 4:5])      # (N, 3, S, S, 1)
                pred_cls_probs = F.softmax(pred[..., 5:], dim=-1)   # (N, 3, S, S, C)
                _, indices = torch.max(pred_cls_probs, dim=-1)      # (N, 3, S, S)
                indices = indices.unsqueeze(-1)                     # (N, 3, S, S, 1)
                pred = torch.cat([ pred[..., :5], indices ], dim=-1)# (N, 3, S, S, 6)
                # Convert coordinate system to normalized format (xywh)
                pboxes = cells_to_boxes(cells=pred, scale=scale)    # (N, 3, S, S, 6)
                tboxes = cells_to_boxes(cells=target, scale=scale)  # (N, 3, S, S, 6)
                #print("tboxes:", tboxes)
                # Filter out bounding boxes from all cells
                for idx, cell_boxes in enumerate(pboxes):
                    obj_mask = cell_boxes[..., 4] > conf_threshold
                    boxes = cell_boxes[obj_mask]
                    pred_bboxes[idx] += boxes.tolist()
                # Filter out bounding boxes from all cells
                for idx, cell_boxes in enumerate(tboxes):
                    obj_mask = cell_boxes[..., 4] > 0.99
                    boxes = cell_boxes[obj_mask]
                    true_bboxes[idx] += boxes.tolist()
            # Perform NMS batch-by-batch
            # =================================================================
            for batch_idx in range(batch_size):
                pbboxes = torch.tensor(pred_bboxes[batch_idx])
                tbboxes = torch.tensor(true_bboxes[batch_idx])
                #print("#############Bboxes before nms#####################")
                #print("Predicted boxes:,", pbboxes)
                #print("Predicted boxes shape:", pbboxes.shape)
                #print("True boxes:", tbboxes)
                #print("True bboxes shape:", tbboxes.shape)
                # Perform NMS class-by-class
                for c in range(n_classes):
                    # Filter pred boxes of specific class
                    nms_pred_boxes = nms_by_class(target=c,
                                                bboxes=pbboxes,
                                                iou_threshold=nms_iou_threshold)
                    nms_true_boxes = nms_by_class(target=c,
                                                bboxes=tbboxes,
                                                iou_threshold=nms_iou_threshold)
                    all_pred_bboxes.extend([[sample_idx]+box
                                            for box in nms_pred_boxes])
                    all_true_bboxes.extend([[sample_idx]+box
                                            for box in nms_true_boxes])
                sample_idx += 1
        # Compute mAP@0.5 & mAP@0.75
        # =================================================================
        # The format of the bboxes is (idx, x1, y1, x2, y2, conf, class)
        #print("all pred booxes:", all_pred_bboxes)
        all_pred_bboxes = torch.tensor(all_pred_bboxes) # (J, 7)
        all_true_bboxes = torch.tensor(all_true_bboxes) # (K, 7)
        print("ALL PRED BBOXES:", all_pred_bboxes)
        print("ALL TRUE BBOXES:", all_true_bboxes)
        print(f"Test number bounding boxes:  predictions:{len(all_pred_bboxes)}  true:{len(all_true_bboxes)}")

        eval50 = mean_average_precision(
                        all_pred_bboxes,
                        all_true_bboxes,
                        iou_threshold=0.5,
                        n_classes=n_classes)
        
        eval75 = mean_average_precision(
                        all_pred_bboxes,
                        all_true_bboxes,
                        iou_threshold=0.75,
                        n_classes=n_classes)
        print((
            f"\t-[mAP@0.5]={eval50['mAP']:.3f}, [Recall]={eval50['recall']:.3f}, [Precision]={eval50['precision']:.3f}\n"
            f"\t-[mAP@0.75]={eval75['mAP']:.3f}, [Recall]={eval75['recall']:.3f}, [Precision]={eval75['precision']:.3f}\n"
            ))
        return eval50['mAP']

check_map(model = model, anchors = anchors, scales = scales, test_loader = test_loader, conf_threshold = 0.9, nms_iou_threshold = 0.3, n_classes = 80)
