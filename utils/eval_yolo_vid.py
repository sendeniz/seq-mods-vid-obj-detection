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

    if args.pretrained is True:
        print("Loading pretrained weights...")

        loaded_checkpoint = torch.load(
            "cpts/yolov4Enas_608_vid_run0_200eps_lambda10.cpt",
        )


    # Load the pretrained model checkpoint 
    model.load_state_dict(loaded_checkpoint["model_state_dict"], strict=False)
    print("Pretrained weights loaded.")

    conf_val = 0.70 #0.57 # use 0.8 for other test
    for epoch in range(args.nepochs):

        len_pred_boxes, len_true_boxes = None, None

        test_loss, test_class_acc, test_noobj_acc, test_obj_acc = testholov4_enas_vid(
            device,
            test_loader,
            model,
            loss_f,
            scaled_anchors,
            conf_thresh=conf_val,
            mode="ciou",
        )

        test_pred_boxes, test_true_boxes = get_bounding_boxes_holo_enas_vid(
            device,
            test_loader,
            model,
            iou_threshold=0.5,
            confidence_threshold=conf_val,
            anchors=anchors,
        )
        test_map, test_recall, test_precision = mean_average_precision(
            device,
            test_pred_boxes,
            test_true_boxes,
            iou_threshold=0.5,
            nclasses=args.nclasses,
        )

    
        len_pred_boxes, len_true_boxes = (
            torch.tensor(len(test_pred_boxes)).to(device),
            torch.tensor(len(test_true_boxes)).to(device),
        )

        print(
            f"Epoch:{epoch + 1}/{args.nepochs} Test[Predictions:{len_pred_boxes if len_pred_boxes is not None else ''} True:{len_true_boxes if len_true_boxes is not None else ''}]"
            )
        print(
            f"Epoch:{epoch + 1}/{args.nepochs} Test[Loss:{test_loss} Class Acc:{test_class_acc} NoObj Acc:{test_noobj_acc} Obj Acc:{test_obj_acc}]"
            )
        print(
            f"Epoch:{epoch + 1}/{args.nepochs} Test[mAP:{test_map if test_map is not None else ''} Precision:{test_precision if test_precision is not None else ''} Recall:{test_recall if test_recall is not None else ''}]"
            )



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
        "--weight_decay", type=float, default=0.0005, help="Weight decay for optimizer"
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
    parser.add_argument(
        "--save_model", type=bool, default=False, help="Store model checkpoints"
    )
    parser.add_argument(
        "--pretrained",
        type=bool,
        default=False,
        help="Use pretrained yolov4 model on mscoco2017",
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
    print(f"  - Save model: {parsed_args.save_model}")
    print(f"  - Pretrained: {parsed_args.pretrained}")

    main(device, args=parsed_args)
