import torch
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import write_results
import numpy as np
from models.yolov6 import YoloV6_EfficentNet
from loss.yolov6loss import YoloV6Loss

from loss.yolov4loss import YoloV4Loss, YoloV4Loss2
from utils.dataset import CoCoDataset
from utils.utils import mean_average_precision, get_bounding_boxes, class_accuracy
from torch.cuda.amp import GradScaler
import os
from torch.utils.tensorboard import SummaryWriter
import argparse

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 640
    path_cpt_file = "cpts/"
    
    # yolo anchors rescaled between 0,1
    # yolo scales and anchors for image size 608, 608
    #S = [19, 38, 76]
    
    # scales for img size 640, 640 
    S = [20, 40, 80]
    anchors = [
                # Large objects (20x20)
                [(0.22, 0.17), (0.30, 0.38), (0.72, 0.63)],
                # Medium (40x40)
                [(0.06, 0.12), (0.12, 0.09), (0.11, 0.23)],
                # Small (80x80) 
                [(0.02, 0.03), (0.03, 0.06), (0.06, 0.04)],
            ]

    writer = SummaryWriter(log_dir="logs/results_yolo/")

    train_dataset = CoCoDataset(
        "data/coco/train_2dot5Kexamples.csv",
        "data/coco/images/",
        "data/coco/labels/",
        #S=S,
        #anchors=anchors,
        image_size=image_size,
        mode="train",
    )

    test_dataset = CoCoDataset(
        "data/coco/test_104examples.csv",
        "data/coco/images/",
        "data/coco/labels/",
        #S=S,
        #anchors=anchors,
        image_size=image_size,
        mode="test",
    )

    scaled_anchors = (
        torch.tensor(anchors) * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(device)
    loss_f = YoloV4Loss2() #YoloV6Loss()

    train_loader = DataLoader(
        dataset=train_dataset,
        num_workers=args.nworkers,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        num_workers=args.nworkers,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )
    
    model = YoloV6_EfficentNet(nclasses=args.nclasses).to(device)
    accumulation_steps = 2
    steps_per_epoch = len(train_loader) // accumulation_steps
    total_steps = args.nepochs * steps_per_epoch
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.momentum, 0.999),
    )
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.nepochs,
        #steps_per_epoch=len(train_loader),
        total_steps=total_steps,
        anneal_strategy="cos",
    )
    scaler = torch.amp.GradScaler()

    for run in range(args.nruns):
        for epoch in range(args.nepochs):
            # Training phase
            model.train()

            
            train_loss, train_class_acc, train_noobj_acc, train_obj_acc = 0, 0, 0, 0
            
            optimizer.zero_grad()  
            for batch_idx, (x, y) in enumerate(train_loader):
                x = x.permute(0, 3, 1, 2).to(device, non_blocking=True)
                y0, y1, y2 = y[0].to(device), y[1].to(device), y[2].to(device)
                
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    preds = model(x)
                    loss = (loss_f(preds[0], y0, scaled_anchors[0]) + 
                           loss_f(preds[1], y1, scaled_anchors[1]) + 
                           loss_f(preds[2], y2, scaled_anchors[2]))
                    
                    loss = loss / accumulation_steps
                
                scaler.scale(loss).backward()
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    # Only step scheduler after the first optimizer step
                    if batch_idx > 0:  
                        scheduler.step()
                
                # Update metrics
                train_loss += loss.item()
                class_acc, noobj_acc, obj_acc = class_accuracy(rank=device,
                    model_pred=preds, 
                    target=y, 
                    confidence_threshold=0.8
                )
                train_class_acc += class_acc
                train_noobj_acc += noobj_acc
                train_obj_acc += obj_acc
            
            # Average metrics over batches
            train_loss /= len(train_loader)
            train_class_acc /= len(train_loader)
            train_noobj_acc /= len(train_loader)
            train_obj_acc /= len(train_loader)
            
            # Testing phase
            model.eval()
            test_loss, test_class_acc, test_noobj_acc, test_obj_acc = 0, 0, 0, 0
            with torch.no_grad():
                for batch_idx, (x, y) in enumerate(test_loader):
                    x = x.permute(0, 3, 1, 2).to(device, non_blocking=True)
                    y0, y1, y2 = y[0].to(device), y[1].to(device), y[2].to(device)
                    
                    with torch.cuda.amp.autocast(dtype=torch.float16):
                        preds = model(x)
                        loss = (loss_f(preds[0], y0, scaled_anchors[0]) + 
                            loss_f(preds[1], y1, scaled_anchors[1]) + 
                            loss_f(preds[2], y2, scaled_anchors[2]))
                    
                    test_loss += loss.item()
                    class_acc, noobj_acc, obj_acc = class_accuracy(rank=device,
                        model_pred=preds, 
                        target=y, 
                        confidence_threshold=0.8
                    )
                    test_class_acc += class_acc
                    test_noobj_acc += noobj_acc
                    test_obj_acc += obj_acc
            
            test_loss /= len(test_loader)
            test_class_acc /= len(test_loader)
            test_noobj_acc /= len(test_loader)
            test_obj_acc /= len(test_loader)
            
            # Compute metrics periodically
            if (epoch + 1 == 1) or ((epoch + 1) % 25 == 0) or (epoch + 1 >= args.nepochs - 1):
                test_pred_boxes, test_true_boxes = get_bounding_boxes(device,
                    test_loader,
                    model,
                    iou_threshold=0.5,
                    confidence_threshold=0.8,
                    anchors=anchors
                )
                
                test_map, test_recall, test_precision = mean_average_precision(device,
                    test_pred_boxes,
                    test_true_boxes,
                    iou_threshold=0.5,
                    nclasses=args.nclasses
                )
                
                print(f"Run:{run+1}/{args.nruns} Epoch:{epoch + 1}/{args.nepochs}")
                print(f"Train[Loss:{train_loss:.4f} Class Acc:{train_class_acc:.4f} NoObj acc:{train_noobj_acc:.4f} Obj Acc:{train_obj_acc:.4f}]")
                print(f"Test[Loss:{test_loss:.4f} Class Acc:{test_class_acc:.4f} NoObj Acc:{test_noobj_acc:.4f} Obj Acc:{test_obj_acc:.4f}]")
                print(f"Metrics[mAP:{test_map:.4f} Precision:{test_precision:.4f} Recall:{test_recall:.4f}]")
                print(f"Detections[Predicted:{len(test_pred_boxes)} True:{len(test_true_boxes)}]")
                
                # Save checkpoint
                if args.save_model:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': train_loss,
                    }, path_cpt_file + f"yolov6_608_run_{run}_epoch_{epoch}.pth")
                    
                    # TensorBoard logging
                    writer.add_scalar('Loss/train', train_loss, epoch)
                    writer.add_scalar('Loss/test', test_loss, epoch)
                    writer.add_scalar('Accuracy/class', train_class_acc, epoch)
                    writer.add_scalar('Accuracy/noobj', train_noobj_acc, epoch)
                    writer.add_scalar('Accuracy/obj', train_obj_acc, epoch)
                    writer.add_scalar('mAP/test', test_map, epoch)
                    writer.add_scalar('Precision/test', test_precision, epoch)
                    writer.add_scalar('Recall/test', test_recall, epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nruns", type=int, default=1, help="Number of runs to perform")
    parser.add_argument("--nepochs", type=int, default=300, help="Number of epochs to perform")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--momentum", type=float, default=0.937, help="Momentum for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay for optimizer")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--nclasses", type=int, default=80, help="Number of classes")
    parser.add_argument("--nworkers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--save_model", action='store_true', default=False, help="Store model checkpoints")
    
    args = parser.parse_args()
    
    print("Training configuration:")
    print(f"- Runs: {args.nruns}")
    print(f"- Epochs: {args.nepochs}")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Learning rate: {args.lr}")
    print(f"- Momentum: {args.momentum}")
    print(f"- Weight decay: {args.weight_decay}")
    print(f"- Workers: {args.nworkers}")
    print(f"- Save models: {args.save_model}")
    
    main(args)