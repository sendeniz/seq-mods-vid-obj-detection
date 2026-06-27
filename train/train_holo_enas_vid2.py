import torch
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.yolo_enas import Yolo_Enas, HippoRNNs
from loss.yolov4loss import YoloV4Loss2, RnnLoss
from utils.dataset_vid import ImageNetVidDataset
from utils.utils import (mean_average_precision, get_bounding_boxes_yolo_enas_vid)
from train_yolo_enas import trainholov4_enas_vid_bptt, testholov4_enas_vid
from torch.cuda.amp import GradScaler
import os
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import datetime
from torch.optim.lr_scheduler import _LRScheduler
from models.rnn import HippoRNN_v3
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device Name: {torch.cuda.get_device_name(device)}" if device.type == 'cuda' else "CPU")
torch.autograd.set_detect_anomaly(True)

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
    writer = SummaryWriter(log_dir="logs/holov4Enas_608_vid_run_100eps")

    train_dataset = ImageNetVidDataset(
        csv_file="data/ILSVRC2015/Data/train_max_80_frames.csv",
        img_dir="data/ILSVRC2015/Data/VID/train/",
        label_dir="data/ILSVRC2015/Data/labels/VID/train/",
        vid_id_csv="data/ILSVRC2015/Data/labels/train_vid_id_max_80_frames.csv",
        frame_ids_dir="data/ILSVRC2015/Data/labels/train_frame_ids/",
        mode="train",
        seq_len=args.seq_len,
    )

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
    loss_yolo_f = YoloV4Loss2(lambda_box=10)
    loss_rnn_f = RnnLoss(lambda_box=10)

    # we drop the last batch to ensure each batch has the same size
    train_loader = DataLoader(
        dataset=train_dataset,
        num_workers=2,
        batch_size=args.batch_size,
        drop_last=False, shuffle = True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        num_workers=2,
        batch_size=args.batch_size,
        drop_last=False, shuffle = False,
    )

   
    for run in range(args.nruns):
        model_yolo = Yolo_Enas(nclasses=args.nclasses).to(device)
        model_rnn =  HippoRNNs(input_size = 1, hidden_size=args.hidden_size, output_size= 3 * 19 * 19 * 1, maxlength=3 * 19 * 19 * 1).to(device)
        if args.pretrained is True:
            print("Loading pretrained weights...")

            loaded_checkpoint = torch.load(
                "cpts/yolov4_608_run_0.cpt",
            )
    
            # remove prediction head layer due to size mismatch
            # caused by coco 80 classes and ilsvcr 30 classes
            keys_to_remove = [
                "yolov4head.0.scaled_pred.1.conv.weight",
                "yolov4head.0.scaled_pred.1.conv.bias",
                "yolov4head.0.scaled_pred.1.bn.weight",
                "yolov4head.0.scaled_pred.1.bn.bias",
                "yolov4head.0.scaled_pred.1.bn.running_mean",
                "yolov4head.0.scaled_pred.1.bn.running_var",
                "yolov4head.1.scaled_pred.1.conv.weight",
                "yolov4head.1.scaled_pred.1.conv.bias",
                "yolov4head.1.scaled_pred.1.bn.weight",
                "yolov4head.1.scaled_pred.1.bn.bias",
                "yolov4head.1.scaled_pred.1.bn.running_mean",
                "yolov4head.1.scaled_pred.1.bn.running_var",
                "yolov4head.2.scaled_pred.1.conv.weight",
                "yolov4head.2.scaled_pred.1.conv.bias",
                "yolov4head.2.scaled_pred.1.bn.weight",
                "yolov4head.2.scaled_pred.1.bn.bias",
                "yolov4head.2.scaled_pred.1.bn.running_mean",
                "yolov4head.2.scaled_pred.1.bn.running_var",
            ]

         
            # Remove the keys from the state dictionary
            for key in keys_to_remove:
                del loaded_checkpoint["model_state_dict"][key]

        # Load the pretrained model checkpoint 
        model_yolo.load_state_dict(loaded_checkpoint["model_state_dict"], strict=False)
        print("Pretrained weights loaded.")

        optimizer_yolo = optim.Adam(
            model_yolo.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.momentum, 0.999),
        )

        optimizer_rnn = optim.Adam(
            model_rnn.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.momentum, 0.999),
        )

        scheduler_yolo = optim.lr_scheduler.OneCycleLR(
            optimizer_yolo,
            max_lr=args.lr,
            epochs=args.nepochs,
            # divide steps per epoch by gradient accumulation steps
            steps_per_epoch=(
                len(train_loader)
                // (args.target_batch_size // (args.batch_size))
            )
            + 1,
            anneal_strategy="cos",
        )

        scheduler_rnn = optim.lr_scheduler.OneCycleLR(
            optimizer_rnn,
            max_lr=args.lr,
            epochs=args.nepochs,
            # divide steps per epoch by gradient accumulation steps
            steps_per_epoch=(
                len(train_loader)
                // (args.target_batch_size // (args.batch_size))
            )
            + 1,
            anneal_strategy="cos",
        )



        for epoch in range(args.nepochs):

            train_map, test_map = None, None
            train_recall, test_recall = None, None
            train_precision, test_precision = None, None
            len_pred_boxes, len_true_boxes = None, None
            # train
            scaler = GradScaler()
            train_loss, train_class_acc, train_noobj_acc, train_obj_acc = (
                trainholov4_enas_vid_bptt(
                    device,
                    train_loader,
                    model_yolo,
                    model_rnn,
                    optimizer_yolo,
                    optimizer_rnn,
                    scheduler_yolo,
                    scheduler_rnn,
                    loss_yolo_f,
                    loss_rnn_f, 
                    scaled_anchors,
                    scaler,
                    conf_thresh=0.8,
                    mode="ciou",
                    target_batch_size=args.target_batch_size,
                    ngpus=args.ngpus,
                )
            )

            test_loss, test_class_acc, test_noobj_acc, test_obj_acc = testholov4_enas_vid(
                device,
                test_loader,
                model_yolo,
                model_rnn,
                loss_yolo_f,
                loss_rnn_f, 
                scaled_anchors,
                conf_thresh=0.8,
                mode="ciou",
            )

            # compute metric at 1st epoch, every 10th epoch and last epoch
            if (
                (epoch + 1 == 1)
                or ((epoch + 1) % 10) == 0
                or (epoch + 1 == args.nepochs)
                or (epoch + 1 == args.nepochs - 1)
                or (epoch == args.nepochs - 1)
            ):
                train_pred_boxes, train_true_boxes = get_bounding_boxes_yolo_enas_vid(
                    device,
                    train_loader,
                    model_yolo,
                    model_rnn,
                    iou_threshold=0.5,
                    confidence_threshold=0.8,
                    anchors=anchors,
                )
                train_map, train_recall, train_precision = mean_average_precision(
                    device,
                    train_pred_boxes,
                    train_true_boxes,
                    iou_threshold=0.5,
                    nclasses=args.nclasses,
                )

                test_pred_boxes, test_true_boxes = get_bounding_boxes_yolo_enas_vid(
                    device,
                    test_loader,
                    model_yolo,
                    model_rnn,
                    iou_threshold=0.5,
                    confidence_threshold=0.8,
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
                f"Run:{run+1}/{args.nruns} Epoch:{epoch + 1}/{args.nepochs} Train[Loss:{train_loss} Class Acc:{train_class_acc} NoObj acc:{train_noobj_acc} Obj Acc:{train_obj_acc}]"
                )
            print(
                f"Run:{run+1}/{args.nruns} Epoch:{epoch + 1}/{args.nepochs} Train[mAP:{train_map if train_map is not None else ''} Precision:{train_precision if train_precision is not None else ''} Recall:{train_recall if train_recall is not None else ''}]"
                )
            print(
                f"Run:{run+1}/{args.nruns} Epoch:{epoch + 1}/{args.nepochs} Test[Predictions:{len_pred_boxes if len_pred_boxes is not None else ''} True:{len_true_boxes if len_true_boxes is not None else ''}]"
                )
            print(
                f"Run:{run+1}/{args.nruns} Epoch:{epoch + 1}/{args.nepochs} Test[Loss:{test_loss} Class Acc:{test_class_acc} NoObj Acc:{test_noobj_acc} Obj Acc:{test_obj_acc}]"
                )
            print(
                f"Run:{run+1}/{args.nruns} Epoch:{epoch + 1}/{args.nepochs} Test[mAP:{test_map if test_map is not None else ''} Precision:{test_precision if test_precision is not None else ''} Recall:{test_recall if test_recall is not None else ''}]"
                )

            if args.save_model and (
                (epoch + 1 == 1)
                or ((epoch + 1) % 10) == 0
                or (epoch + 1 == args.nepochs)
            ):
                torch.save(
                    {
                        "epoch": epoch,
                        "run": run,
                        "model_state_dict": model_yolo.state_dict(),
                        "optimizer_state_dict": optimizer_yolo.state_dict(),
                        "scheduler_state_dict": scheduler_yolo.state_dict(),
                    },
                    path_cpt_file + f"YoloEnas_608_vid_{run}_new.cpt",
                )

                torch.save(
                    {
                        "epoch": epoch,
                        "run": run,
                        "model_state_dict": model_rnn.state_dict(),
                        "optimizer_state_dict": optimizer_rnn.state_dict(),
                        "scheduler_state_dict": scheduler_rnn.state_dict(),
                    },
                    path_cpt_file + f"YoloEnas_608_vid_{run}_new.cpt",
                )

                # train logs
                writer.add_scalar(f"Loss/train/run_{run}", train_loss, epoch)
                writer.add_scalar(
                    f"Class Acc/train/run_{run}", train_class_acc, epoch
                )
                writer.add_scalar(
                    f"No Obj Acc/train/run_{run}", train_noobj_acc, epoch
                )
                writer.add_scalar(f"Obj Acc/train/run_{run}", train_obj_acc, epoch)

                writer.add_scalar(f"Map/train/run_{run}", train_map, epoch)
                writer.add_scalar(
                    f"Precision/train/run_{run}", train_precision, epoch
                )
                writer.add_scalar(f"Recall/train/run_{run}", train_recall, epoch)

                # test logs
                writer.add_scalar(f"Loss/test/run_{run}", test_loss, epoch)
                writer.add_scalar(
                    f"Class Acc/test/run_{run}", test_class_acc, epoch
                )
                writer.add_scalar(
                    f"No Obj Acc/test/run_{run}", test_noobj_acc, epoch
                )
                writer.add_scalar(f"Obj Acc/test/run_{run}", test_obj_acc, epoch)

                writer.add_scalar(f"Map/test/run_{run}", test_map, epoch)
                writer.add_scalar(
                    f"Precision/test/run_{run}", test_precision, epoch
                )
                writer.add_scalar(f"Recall/test/run_{run}", test_recall, epoch)

                print(
                    f"Checkpoint and evaluation from run:{run+1} and epoch {epoch + 1} stored"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nruns", type=int, default=1, help="Number of runs to perform"
    )
    parser.add_argument(
        "--nepochs", type=int, default=50, help="Number of epochs to perform"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for training"
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
    parser.add_argument(
        "--ngpus", type=int, default=1, help="Number of gpus used for training"
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=32,
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
    print("Yolo Enas and Hippo RNN inistalised with following paremters:")

    print(f"  - Number of runs: {parsed_args.nruns}")
    print(f"  - Number of epochs: {parsed_args.nepochs}")
    print(f"  - Batch size: {parsed_args.batch_size}")
    print(f"  - Target batch size: {parsed_args.target_batch_size}")
    print(f"  - Sequence length: {parsed_args.seq_len}")
    print(f"  - Number of classes: {parsed_args.nclasses}")
    print(f"  - Learning rate: {parsed_args.lr}")
    print(f"  - Hidden size: {parsed_args.hidden_size}")
    print(f"  - Momentum: {parsed_args.momentum}")
    print(f"  - Weight decay: {parsed_args.weight_decay}")
    print(f"  - Number of gpus: {parsed_args.ngpus}")
    print(f"  - Save model: {parsed_args.save_model}")
    print(f"  - Pretrained: {parsed_args.pretrained}")

    main(device, args=parsed_args)
