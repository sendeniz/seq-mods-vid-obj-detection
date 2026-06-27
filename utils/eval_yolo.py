import torch
from utils.dataset import CoCoDataset
from utils.utils import get_bouding_boxes, mean_average_precision
from torch.utils.data import DataLoader
from models.yolov4 import YoloV4_EfficentNet
import torch.optim as optim

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
S = [19, 38, 76]
anchors = [
       [(0.23, 0.18), (0.32, 0.40), (0.75, 0.66)],
       [(0.06, 0.12), (0.12, 0.09), (0.12, 0.24)],
       [(0.02, 0.03), (0.03, 0.06), (0.07, 0.05)],
       ]

conf_thresh = 0.8 #0.75 #0.7 #0.6 #0.5 #0.45 #0.5
nms_iou_thresh = 0.5
map_iou_thresh = 0.5
batch_size = 32
nworkers = 2
nclasses = 80
lr =  0.00001
weight_decay = 0.0005
path_cpt_file = f'cpts/yolov4_608_mscoco.cpt'
checkpoint = torch.load(path_cpt_file)
model = YoloV4_EfficentNet(nclasses = nclasses).to(device)
optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
model.eval()
print("Petrained YoloV4 608 EfficentNet S Net initalized.")

test_dataset = test_dataset = CoCoDataset("data/coco/test.csv", "data/coco/images/", "data/coco/labels/",
                          S = S, anchors = anchors, mode = 'test')

test_loader = DataLoader(dataset = test_dataset, num_workers = nworkers,
                                            batch_size = batch_size,
                                            shuffle = False, drop_last = False)


test_pred_boxes, test_true_boxes = get_bouding_boxes(test_loader, model, iou_threshold = nms_iou_thresh, confidence_threshold = conf_thresh, anchors = anchors)
print(f"Test number bounding boxes:  predictions:{len(test_pred_boxes)}  true:{len(test_true_boxes)}")
test_map_val, test_recall_val, test_precision_val = mean_average_precision(test_pred_boxes, test_true_boxes, map_iou_thresh, nclasses = nclasses)
print(f"Test mAP:{test_map_val}")
print(f"Test recall:{test_recall_val}")
print(f"Test precision:{test_precision_val}")


