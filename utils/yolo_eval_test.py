import torch
from utils.dataset import CoCoDataset
from torch.utils.data import DataLoader
from models.yolov4 import YoloV4_EfficentNet
import torch.optim as optim
from collections import Counter
import time
torch.set_grad_enabled(False)
device = torch.device('cuda:4' if torch.cuda.is_available() else 'cpu')
S = [13, 26, 52]
anchors = [
       [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
       [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
       [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
       ]


conf_thresh = 0.5 #0.45 #0.5
nms_iou_thresh = 0.5
map_iou_thresh = 0.5
batch_size = 64
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
model.eval()
print("Petrained YoloV4 416 EfficentNet S Net initalized.")

test_dataset = test_dataset = CoCoDataset("data/coco/test_208examples.csv", "data/coco/images/", "data/coco/labels/",
                          S = S, anchors = anchors, mode = 'test')

test_loader = DataLoader(dataset = test_dataset, num_workers = nworkers,
                                            batch_size = batch_size,
                                            shuffle = False, drop_last = False)



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
    #start_time = time.time()
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
    #end_time = time.time()
    #print(f"Box intersection over union calculation: DONE. Execution time: {end_time - start_time} seconds")
    if mode == 'iou':
        return iou

    if mode == 'giou':
        # Convex diagnal length
        convex_w = torch.max(box1_x2, box2_x2) - torch.min(box1_x1, box2_x1)
        convex_h = torch.max(box1_y2, box2_y2) - torch.min(box1_y1, box2_y1)
        convex_area = convex_w * convex_h + epsilon
        giou = iou - ((convex_area - union) / convex_area)
        return giou

    if mode == 'ciou':
        convex_w = torch.max(box1_x2, box2_x2)-torch.min(box1_x1, box2_x1)
        convex_h = torch.max(box1_y2, box2_y2)-torch.min(box1_y1, box2_y1)
        c2 = convex_w ** 2 + convex_h ** 2 + epsilon  # convex diagonal squared
        rho2 = ((box2_x1 + box2_x2 - box1_x1 - box1_x2) ** 2 +
                (box2_y1 + box2_y2 - box1_y1 - box1_y2) ** 2) / 4  # center distance squared
        # some of the MsCoCo 2017 annotations before or after augmentations seem to have a box height of 0.0
        # this causes NaN values in loss due to dvision by zero in the ciou computation
        # for stability we add a small constant epsilon to box2_h and box1_1 above
        v = (4 / torch.pi ** 2) * torch.pow(torch.atan(box2_w / box2_h) - torch.atan(box1_w / box1_h), 2)
        with torch.no_grad():
            alpha = v / ((1 + epsilon) - iou + v)
        ciou = iou - (rho2 / c2 + v * alpha)
        return ciou

def save_list_to_txt(lst, filename):
    with open(filename, 'w') as file:
        for row in lst:
            file.write(' '.join(map(str, row)) + '\n')


def mean_average_precision(
    pred_boxes, true_boxes, iou_threshold=0.5, nclasses=80, mode = 'iou'):
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
    # iterate over classes category
    avg_precision = []
    save_list_to_txt(pred_boxes, 'test_pred_boxes.txt')
    save_list_to_txt(true_boxes, 'test_true_boxes.txt')
    pred_boxes = torch.tensor(pred_boxes)
    true_boxes = torch.tensor(true_boxes)

    for c in range(nclasses):
        # init candidate detections and ground truth as an empty list for storage

        candidate_detections = []
        ground_truths = []

        # iterate over candidate bouding box predictions
        # index 1 is the class prediction and if equal to class c we are currently
        # looking at, then we append
        # if the candidate detection in the bounding box predictions is equal
        # to the class category c we are currently looking at add it to candidate list

        candidate_detections = pred_boxes[pred_boxes[..., 1] == c] #.tolist()
        ground_truths = true_boxes[true_boxes[..., 1] == c] #.tolist()
        total_true_bboxes = ground_truths.size(0) #len(ground_truths)
        total_pred_bboxes = candidate_detections.size(0) #len(candidate_detections)
        if total_true_bboxes == 0 or total_pred_bboxes == 0:
            continue
        # first index 0 is the training index, given image zero with 3 bbox
        # and img 1 has 5 bounding boxes, Counter will count how many bboxes
        # and create a dictionary, so amoung_bbox = [0:3, 1:5]
        amount_bboxes = Counter([gt[0].item() for gt in ground_truths])
        # go through each key, val in dic
        # and convert: i.e., example below
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)
        order = candidate_detections[..., 2].argsort(descending = True)
        candidate_detections = candidate_detections[order].tolist()
        ground_truths = ground_truths #.tolist()
        # sort over probability scores

        # length for true positives and false positives for class based on detection
        # initalise tensors of zeros for true positives (TP) and false positives
        # (FP) as the length of possible candidate detections for a given class C
        TP = torch.zeros((len(candidate_detections)))
        FP = torch.zeros((len(candidate_detections)))

        for detection_idx, detection in enumerate(candidate_detections):

            # use only the ground_truths that have the same training idx as detection
            ground_truth_img = ground_truths[ground_truths[:, 0] == detection[0]].tolist()
            num_gts = len(ground_truth_img)
            best_iou = torch.zeros(num_gts)
            num_detections = len(detection)
            if num_gts == 0 or num_detections == 0:
                continue

            # iterate over all ground truth bbox in grout truth image
            for idx, gt in enumerate(ground_truth_img):
                iou = box_intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]), mode = mode
                )

                best_iou[idx] = iou.max()
            best_iou, best_gt_idx = best_iou.max(dim=0)
            best_iou = best_iou.item()
            best_gt_idx = best_gt_idx.item()

            if best_iou > iou_threshold:
                # check if the bounding box has already used for detection
                # set it to 1 since we already covered the bounding box
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                # else if the iou was not greater than the treshhold set as FP
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1
        # compute average precision by integrating using numeric integration
        # with the trapozoid method starting at point x = 1, y = 0
        # starting points are added to precision = x and recall = y using
        # torch cat
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + 1e-6)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + 1e-6)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        avg_precision.append(torch.trapz(precisions, recalls))

    return sum(avg_precision) / len(avg_precision)

def non_max_suppression(boxes, iou_threshold, confidence_threshold):
    """Non-maximal surpression on yolo bounding boxes using intersection over union (IoU)

    Arguments:
        boxes (tensor): tensor of shape (N, 6) in yolo format (class, confidence score, x, y, w, h)
        iou_threshold (float): minimum valid box confidence threshold

    Returns:
        boxes (tensor): tensor of shape (N, 6)

    """
    boxes = torch.tensor(boxes)
    if boxes.size(0) <= 1:
        return boxes.tolist()

    # create confidence mask to filter out bounding boxes that are below confidence threshold
    confidence_mask = boxes[:, 1] >= confidence_threshold
    boxes =  boxes[confidence_mask]
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
    #print("Converting cells to bboxes")
    #start_time = time.time()
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
    #end_time = time.time()
    #print(f"Converting cells to bboxes: DONE. Execution time: {end_time - start_time} seconds")
    return converted_bboxes.tolist()

def get_bouding_boxes(loader, model, iou_threshold, anchors, conf_threshold,
    mode = 'iou'):
    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, (x, labels) in enumerate(loader):
        if (batch_idx + 1) % 1 == 0:
            print(f"Fetching bounding boxes from batch: {batch_idx + 1} / {len(loader)}.")
        x = x.permute(0, 3, 1, 2)
        x = x.to(torch.float32).to(device)
        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        bboxes = [[] for _ in range(batch_size)]
        for i in range(3):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(device) * S
            boxes_scale_i = cells_to_bboxes(
                predictions[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box
        # we just want one bbox for each label, not one for each scale
        true_bboxes = cells_to_bboxes(
            labels[2], anchor, S=S, is_preds=False
        )
        #start_time = time.time()
        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                confidence_threshold=conf_threshold
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)
            for box in true_bboxes[idx]:
                if box[1] > conf_threshold:
                    all_true_boxes.append([train_idx] + box)
            train_idx += 1
        #end_time = time.time()
        #print(f"Get booxes NMS Process: DONE. Execution time: {end_time - start_time} seconds")
    model.train()
    return all_pred_boxes, all_true_boxes

test_pred_boxes, test_true_boxes = get_bouding_boxes(test_loader, model, iou_threshold = nms_iou_thresh, conf_threshold = conf_thresh, anchors = anchors, mode = 'iou')
print(f"Test number bounding boxes:  predictions:{len(test_pred_boxes)}  true:{len(test_true_boxes)}")
test_map_val = mean_average_precision(test_pred_boxes, test_true_boxes, map_iou_thresh, nclasses = nclasses, mode = 'iou')
print(f"Test mAP:{test_map_val}")
