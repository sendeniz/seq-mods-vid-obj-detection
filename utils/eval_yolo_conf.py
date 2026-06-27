import torch
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.yolov4 import YoloV4_EfficentNet
from loss.yolov4loss import YoloV4Loss2
from utils.dataset import CoCoDataset
from utils.utils import (mean_average_precision, get_bounding_boxes, class_accuracy)
from train.holov4_enas_vid_train_fn import trainholov4_enas_vid_bptt, testholov4_enas_vid
from torch.cuda.amp import GradScaler
import os
import torch.optim as optim
import argparse
import datetime

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device Name: {torch.cuda.get_device_name(device)}" if device.type == 'cuda' else "CPU")


def main(args):

    image_size = 608
    path_cpt_file = "cpts/"

    # yolo anchors rescaled between 0,1
    # yolo scales and anchors for image size 608, 608
    S = [19, 38, 76]
    anchors = [
        [(0.23, 0.18), (0.32, 0.40), (0.75, 0.66)],
        [(0.06, 0.12), (0.12, 0.09), (0.12, 0.24)],
        [(0.02, 0.03), (0.03, 0.06), (0.07, 0.05)],
    ]
    
    test_dataset = CoCoDataset(
            "data/coco/test.csv",
            "data/coco/images/",
            "data/coco/labels/",
            #S=S,
            #anchors=anchors,
            image_size=image_size,
            mode="test",
        )

    scaled_anchors = (
        torch.tensor(anchors)
        * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(device)
    loss_f = YoloV4Loss2(lambda_box=10)

    test_loader = DataLoader(
            dataset=test_dataset,
            num_workers=args.nworkers,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

    model = YoloV4_EfficentNet(nclasses=args.nclasses, gate=args.gate).to(device)
    
    for c in range(0,1):
        # Initialize lists to store the results
        confidence_values = []
        num_predictions = []
        mAP_values = []
        print("Loading pretrained weights...")
        # name = f"yolov4_s4nd_v4_run_{c}_best_map"
        name = f"yolov4_run_{c}_best_map"
        print(f"Checkpoint file: {name}")
        loaded_checkpoint = torch.load(
            f"ckpts/{name}.pth",
        )

        # Load the pretrained model checkpoint 
        model.load_state_dict(loaded_checkpoint["model_state_dict"], strict=False)
        print("Pretrained weights loaded.")
        model.eval()
        for epoch in range(args.nepochs):
            # Iterate through confidence values from 0.0 to 1.0 in steps of 0.01
            # for confidence in [round(i * 0.01, 2) for i in range(0, 101)]:
            #for confidence in [0.8095]: # s4nd run 0
            #for confidence in [0.82425]: # yolov4 run 0 
            for confidence in [0.8]: # s4nd run 0
                print("Confidence:", confidence)
                # Obtain predictions and calculate metrics
                test_pred_boxes, test_true_boxes = get_bounding_boxes(device,
                    test_loader,
                    model,
                    iou_threshold=0.35,
                    confidence_threshold=confidence,
                    anchors=anchors
                )
                print(f"N test preds vs true preds: {len(test_pred_boxes)} / {len(test_true_boxes)}")
                # Calculate mAP, recall, and precision
                test_map_05, test_recall_05, test_precision_05 = mean_average_precision(
                    device,
                    test_pred_boxes,
                    test_true_boxes,
                    iou_threshold=0.5,
                    nclasses=args.nclasses,
                )
                test_map_075, test_recall_075, test_precision_075 = mean_average_precision(
                    device,
                    test_pred_boxes,
                    test_true_boxes,
                    iou_threshold=0.75,
                    nclasses=args.nclasses,
                )
                test_map_095, test_recall_095, test_precision_095 = mean_average_precision(
                    device,
                    test_pred_boxes,
                    test_true_boxes,
                    iou_threshold=0.95,
                    nclasses=args.nclasses,
                )
                print(f"mAP@0.5:{test_map_05} - recall@0.5:{test_recall_05} - precision@0.5:{test_precision_05} ")
                print(f"mAP@0.75:{test_map_075} - recall@0.75:{test_recall_075} - precision@0.75:{test_precision_075} ")
                print(f"mAP@0.95:{test_map_095} - recall@0.95:{test_recall_095} - precision@0.95:{test_precision_095} ")
                print(f"Average mAP:{(test_map_05 + test_map_075 + test_map_095) / 3:.4f}")
                print(f"Average recall:{(test_recall_05 + test_recall_075 + test_recall_095) / 3:.4f}")
                print(f"Average precision:{(test_precision_05 + test_precision_075 + test_precision_095) / 3:.4f}")
                # Store the results
                confidence_values.append(confidence)
                num_predictions.append(len(test_pred_boxes))
                mAP_values.append(test_map_05)
        # Save the results to a .txt file
        # Save confidence values to a separate file
        #with open(f"{name}_conf.txt", "w") as f:
        #   #f.write("Confidence\n")  # Header
        #    for conf in confidence_values:
        #        f.write(f"{conf}\n")  # Write each confidence value

        # Save number of predictions to a separate file
        #with open(f"{name}_num_preds.txt", "w") as f:
        #    #f.write("Number of Predictions\n")  # Header
        #    for num_pred in num_predictions:
        #        f.write(f"{num_pred}\n")  # Write each number of predictions

        # Save mAP values to a separate file
        #with open(f"{name}_map.txt", "w") as f:
        #    #f.write("mAP\n")  # Header
        #    for mAP in mAP_values:
        #        f.write(f"{mAP}\n")  # Write each mAP value

        #print("Results saved to results.txt")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--nepochs", type=int, default=1, help="Eval epoch 1.")
    parser.add_argument("--momentum", type=float, default=0.937, help="Momentum for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay for optimizer")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--nclasses", type=int, default=80, help="Number of classes")
    parser.add_argument("--nworkers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--gate", type=str, default=None, help="Recurrence for global context.")

    args = parser.parse_args()
    
    print("Eval configuration:")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Epochs: {args.nepochs}")
    print(f"- Learning rate: {args.lr}")
    print(f"- Momentum: {args.momentum}")
    print(f"- Weight decay: {args.weight_decay}")
    print(f"- Workers: {args.nworkers}")
    
    main(args)
