import torch
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.holov4_enas import HoloV4_Enas_EfficentNet
from loss.yolov4loss import YoloV4Loss2
from utils.dataset_vid import ImageNetVidDataset
from utils.utils import (mean_average_precision, get_bounding_boxes_holo_enas_vid)
from train.holov4_enas_vid_train_fn import trainholov4_enas_vid_bptt, testholov4_enas_vid
from torch.cuda.amp import GradScaler
import os
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import datetime
from torch.optim.lr_scheduler import _LRScheduler

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device Name: {torch.cuda.get_device_name(device)}" if device.type == 'cuda' else "CPU")


def main(device, args):

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
    
    test_dataset = test_dataset = ImageNetVidDataset(
        csv_file="data/ILSVRC2015/Data/val_max_80_frames.csv",
        img_dir="data/ILSVRC2015/Data/VID/val/",
        label_dir="data/ILSVRC2015/Data/labels/VID/val/",
        vid_id_csv="data/ILSVRC2015/Data/labels/val_vid_id_max_80_frames.csv",
        frame_ids_dir="data/ILSVRC2015/Data/labels/val_frame_ids/",
        mode="test",
        seq_len=args.seq_len,
    )

    scaled_anchors = (
        torch.tensor(anchors)
        * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(device)
    loss_f = YoloV4Loss2(lambda_box=10)

    test_loader = DataLoader(
        dataset=test_dataset,
        num_workers=4,
        batch_size=args.batch_size,
        drop_last=False, shuffle = False,
    )

    model = HoloV4_Enas_EfficentNet(hidden_size=args.hidden_size, maxlength=args.seq_len, nclasses=args.nclasses, gate = args.gate,
    ).to(device)
    
    for c in range(0,1):
        # Initialize lists to store the results
        confidence_values = []
        num_predictions = []
        mAP_values = []
        print("Loading pretrained weights...")
        name = f"s4ndv3yolov4Enas_608_vid_run{c}_200eps_lambda10"
        print(f"Checkpoint file: {name}")
        loaded_checkpoint = torch.load(
            f"cpts/{name}.cpt",
        )

        # Load the pretrained model checkpoint 
        model.load_state_dict(loaded_checkpoint["model_state_dict"], strict=False)
        print("Pretrained weights loaded.")
        model.eval()
        for epoch in range(args.nepochs):
            # Iterate through confidence values from 0.0 to 1.0 in steps of 0.01
            for confidence in [round(i * 0.01, 2) for i in range(0, 101)]:
                print("Confidence:", confidence)
                # Obtain predictions and calculate metrics
                test_pred_boxes, test_true_boxes = get_bounding_boxes_holo_enas_vid(
                    device,
                    test_loader,
                    model,
                    iou_threshold=0.5,
                    confidence_threshold=confidence,  # Use the current confidence
                    anchors=anchors,
                )
                print(f"N test preds vs true preds: {len(test_pred_boxes)} / {len(test_true_boxes)}")
                # Calculate mAP, recall, and precision
                test_map, test_recall, test_precision = mean_average_precision(
                    device,
                    test_pred_boxes,
                    test_true_boxes,
                    iou_threshold=0.5,
                    nclasses=args.nclasses,
                )
                print(f"mAP@0.5:{test_map}")

                # Store the results
                confidence_values.append(confidence)
                num_predictions.append(len(test_pred_boxes))
                mAP_values.append(test_map)
        # Save the results to a .txt file
        # Save confidence values to a separate file
        with open(f"{name}_conf.txt", "w") as f:
            #f.write("Confidence\n")  # Header
            for conf in confidence_values:
                f.write(f"{conf}\n")  # Write each confidence value

        # Save number of predictions to a separate file
        with open(f"{name}_num_preds.txt", "w") as f:
            #f.write("Number of Predictions\n")  # Header
            for num_pred in num_predictions:
                f.write(f"{num_pred}\n")  # Write each number of predictions

        # Save mAP values to a separate file
        with open(f"{name}_map.txt", "w") as f:
            #f.write("mAP\n")  # Header
            for mAP in mAP_values:
                f.write(f"{mAP}\n")  # Write each mAP value

        print("Results saved to results.txt")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nruns", type=int, default=1, help="Number of runs to perform"
    )
    parser.add_argument(
        "--nepochs", type=int, default=1, help="Number of epochs to perform"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--target_batch_size",
        type=int,
        default=128,
        help="Target batch size for training",
    )
    parser.add_argument(
        "--momentum", type=float, default=0.937, help="Momentum for optimizer"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0000, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--hidden_size", type=int, default=1024, help="Hidden size used for Hippo Rnn"
    )
    parser.add_argument("--nclasses", type=int, default=30, help="Number of classes")
    parser.add_argument("--gate", type=str, default="urlstm", help="Rnn gate: pick urlstm or hippolstm.")
    parser.add_argument(
        "--ngpus", type=int, default=1, help="Number of gpus used for training"
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=8,
        help="Number of frames from a video that form a sequence",
    )

    parsed_args = parser.parse_args()
    if parsed_args.gate == "hippolstm":
        print("HoloV4 Enas HippoLstm: initialized with the following parameters:")
    elif parsed_args.gate == "urlstm":
        print("HoloV4 Enas URLstm: initialized with the following parameters:")
    elif parsed_args.gate == "linear":
        print("HoloV4 Enas Linear: initialized with the following parameters:")
    elif parsed_args.gate == "none":
        print("HoloV4 Enas No gateing: initialized with the following parameters:")

    print(f"  - Number of runs: {parsed_args.nruns}")
    print(f"  - Number of epochs: {parsed_args.nepochs}")
    print(f"  - Batch size: {parsed_args.batch_size}")
    print(f"  - Target batch size: {parsed_args.target_batch_size}")
    print(f"  - Sequence length: {parsed_args.seq_len}")
    print(f"  - Number of classes: {parsed_args.nclasses}")
    print(f"  - Rnn gate: {parsed_args.gate}")
    print(f"  - Learning rate: {parsed_args.lr}")
    print(f"  - Hidden size: {parsed_args.hidden_size}")
    print(f"  - Momentum: {parsed_args.momentum}")
    print(f"  - Weight decay: {parsed_args.weight_decay}")
    print(f"  - Number of gpus: {parsed_args.ngpus}")

    main(device, args=parsed_args)
