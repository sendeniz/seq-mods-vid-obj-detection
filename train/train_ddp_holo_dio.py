import torch
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import write_results
import numpy as np
from models.yolov4 import YoloV4_EfficentNet
from loss.yolov4loss import YoloV4Loss, YoloV4Loss2
from utils.dataset import CoCoDataset
from utils.utils import mean_average_precision, get_bounding_boxes, class_accuracy
from yolov4_train_fn import trainyolov4, testyolov4
from torch.cuda.amp import GradScaler
import os
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import argparse


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12345"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size, args):
    image_size = 608
    path_cpt_file = "cpts/"
    setup(rank, world_size)
    # yolo anchors rescaled between 0,1
    # yolo scales and anchors for image size 608, 608
    S = [19, 38, 76]
    anchors = [
        [(0.23, 0.18), (0.32, 0.40), (0.75, 0.66)],
        [(0.06, 0.12), (0.12, 0.09), (0.12, 0.24)],
        [(0.02, 0.03), (0.03, 0.06), (0.07, 0.05)],
    ]
    writer = SummaryWriter(log_dir="logs/results_yolo/")

    train_dataset = CoCoDataset(
        "data/coco/train.csv",
        "data/coco/images/",
        "data/coco/labels/",
        S=S,
        anchors=anchors,
        image_size=image_size,
        mode="train",
    )

    test_dataset = test_dataset = CoCoDataset(
        "data/coco/test.csv",
        "data/coco/images/",
        "data/coco/labels/",
        S=S,
        anchors=anchors,
        image_size=image_size,
        mode="test",
    )

    scaled_anchors = (
        torch.tensor(anchors)
        * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(rank)
    loss_f = YoloV4Loss2()

    sampler_train = DistributedSampler(train_dataset, shuffle=True)
    sampler_test = DistributedSampler(test_dataset, shuffle=False)

    # we drop the last batch to ensure each batch has the same size
    train_loader = DataLoader(
        dataset=train_dataset,
        num_workers=4,
        batch_size=args.batch_size,
        drop_last=False,
        sampler=sampler_train,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        num_workers=4,
        batch_size=args.batch_size,
        drop_last=False,
        sampler=sampler_test,
    )
    model = YoloV4_EfficentNet(nclasses=args.nclasses).to(rank)

    ddp_model = DDP(model, device_ids=[rank], gradient_as_bucket_view=True)
    optimizer = optim.Adam(
        ddp_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.momentum, 0.999),
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.nepochs,
        steps_per_epoch=len(train_loader),
        anneal_strategy="cos",
    )
    print(f"Rank:{rank} initalised")
    world_size = dist.get_world_size()
    for run in range(args.nruns):
        for epoch in range(args.nepochs):
            train_map, test_map = None, None
            train_recall, test_recall = None, None
            train_precision, test_precision = None, None
            len_pred_boxes, len_true_boxes = None, None
            # train
            scaler = GradScaler()
            train_loss, train_class_acc, train_noobj_acc, train_obj_acc = trainyolov4(
                rank,
                train_loader,
                model,
                optimizer,
                scheduler,
                loss_f,
                scaled_anchors,
                scaler,
                conf_thresh=0.8,
                mode="ciou",
            )

            # Perform all-reduce only on the master process (rank 0)
            dist.all_reduce(train_loss)
            dist.all_reduce(train_class_acc)
            dist.all_reduce(train_noobj_acc)
            dist.all_reduce(train_obj_acc)

            # Divide by the world size to compute the average
            train_loss /= world_size
            train_class_acc /= world_size
            train_noobj_acc /= world_size
            train_obj_acc /= world_size

            test_loss, test_class_acc, test_noobj_acc, test_obj_acc = testyolov4(
                rank,
                test_loader,
                model,
                loss_f,
                scaled_anchors,
                conf_thresh=0.8,
                mode="ciou",
            )

            # Perform all-reduce only on the master process (rank 0)
            dist.all_reduce(test_loss)
            dist.all_reduce(test_class_acc)
            dist.all_reduce(test_noobj_acc)
            dist.all_reduce(test_obj_acc)

            # Divide by the world size to compute the average
            test_loss /= world_size
            test_class_acc /= world_size
            test_noobj_acc /= world_size
            test_obj_acc /= world_size

            dist.barrier()
            # compute metric at 1st epoch, every 10th epoch and last epoch
            if (
                (epoch + 1 == 1)
                or ((epoch + 1) % 10) == 0
                or (epoch + 1 == args.nepochs)
                or (epoch + 1 == args.nepochs - 1)
                or (epoch == args.nepochs - 1)
            ):
                test_pred_boxes, test_true_boxes = get_bounding_boxes(
                    rank,
                    test_loader,
                    model,
                    iou_threshold=0.5,
                    confidence_threshold=0.8,
                    anchors=anchors,
                )
                test_map, test_recall, test_precision = mean_average_precision(
                    rank,
                    test_pred_boxes,
                    test_true_boxes,
                    iou_threshold=0.5,
                    nclasses=args.nclasses,
                )
                len_pred_boxes, len_true_boxes = (
                    torch.tensor(len(test_pred_boxes)).to(rank),
                    torch.tensor(len(test_true_boxes)).to(rank),
                )

                dist.all_reduce(len_pred_boxes)
                dist.all_reduce(len_true_boxes)

                dist.all_reduce(test_map)
                dist.all_reduce(test_recall)
                dist.all_reduce(test_precision)
                test_map /= world_size
                test_recall /= world_size
                test_precision /= world_size

            if dist.get_rank() == 0:
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

            # ddp throws a timeout error at last epoch so we add additional save model option
            # to also save before the last epoch and an additional barrier
            dist.barrier()
            if args.save_model and (
                (epoch + 1 == 1)
                or ((epoch + 1) % 10) == 0
                or (epoch + 1 == args.nepochs)
                or (epoch + 1 == args.nepochs - 1)
                or (epoch == args.nepochs - 1)
            ):
                dist.barrier()
                if dist.get_rank() == 0:
                    torch.save(
                        {
                            "epoch": epoch,
                            "run": run,
                            "model_state_dict": ddp_model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                        },
                        path_cpt_file + f"yolov4_608_run_{run}.cpt",
                    )

                    writer.add_scalar("Loss/train", train_loss, epoch)
                    writer.add_scalar("Class Acc/train", train_class_acc, epoch)
                    writer.add_scalar("No Obj Acc/train", train_noobj_acc, epoch)
                    writer.add_scalar("Obj Acc/train", train_obj_acc, epoch)

                    writer.add_scalar("Loss/test", test_loss, epoch)
                    writer.add_scalar("Class Acc/test", test_class_acc, epoch)
                    writer.add_scalar("No Obj Acc/test", test_noobj_acc, epoch)
                    writer.add_scalar("Obj Acc/test", test_obj_acc, epoch)

                    writer.add_scalar("Map/test", test_map, epoch)
                    writer.add_scalar("Precision/test", test_precision, epoch)
                    writer.add_scalar("Recall/test", test_recall, epoch)

                    print(f"Checkpoint and evaluation at epoch {epoch + 1} stored")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nruns", type=int, default=1, help="Number of runs to perform"
    )
    parser.add_argument(
        "--nepochs", type=int, default=300, help="Number of epochs to perform"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
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
    parser.add_argument("--nclasses", type=int, default=80, help="Number of classes")
    parser.add_argument(
        "--ngpus", type=int, default=1, help="Number of gpus used for training"
    )
    parser.add_argument(
        "--save_model", type=bool, default=False, help="Store model checkpoints"
    )

    parsed_args = parser.parse_args()
    print("YoloV4 initialized with the following parameters:")
    print(f"  - Number of runs: {parsed_args.nruns}")
    print(f"  - Number of epochs: {parsed_args.nepochs}")
    print(f"  - Batch size: {parsed_args.batch_size}")
    print(f"  - Target batch size: {parsed_args.target_batch_size}")
    print(f"  - Learning rate: {parsed_args.lr}")
    print(f"  - Momentum: {parsed_args.momentum}")
    print(f"  - Weight decay: {parsed_args.weight_decay}")
    print(f"  - Number of gpus: {parsed_args.ngpus}")

    # pass ngpus as rank and nprocs
    mp.spawn(
        main,
        args=(parsed_args.ngpus, parsed_args),
        nprocs=parsed_args.ngpus,
        join=True,
    )
