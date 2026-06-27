import torch
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import write_results
import numpy as np
from models.yolov4 import YoloV4_EfficientNet
from models.yolov5 import YoloV5_EfficientNet
from loss.yolov4loss import YoloV4Loss, YoloV4Loss2
from loss.yolov5loss import YoloV5Loss 
from utils.dataset import CoCoDataset
from utils.utils import mean_average_precision, get_bounding_boxes, class_accuracy
from utils.ema import EMA
from torch.cuda.amp import GradScaler
import os
from torch.utils.tensorboard import SummaryWriter
import argparse
import random
from tqdm import tqdm
import time



def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_size = 640

    if img_size == 640:
        anchors = [
        # Large objects (assigned to smallest grid, e.g., 20x20)
        [(0.22, 0.17), (0.30, 0.38), (0.72, 0.63)],
        # Medium objects (e.g., 40x40)
        [(0.06, 0.12), (0.12, 0.09), (0.11, 0.23)],
        # Small objects (e.g., 80x80)
        [(0.02, 0.03), (0.03, 0.06), (0.06, 0.04)],
        ]
        # Grid sizes (640/32=20, 640/16=40, 640/8=80)
        S = [20, 40, 80]  

    if img_size == 608:
        anchors = [
        [(0.23, 0.18), (0.32, 0.40), (0.75, 0.66)],
        [(0.06, 0.12), (0.12, 0.09), (0.12, 0.24)],
        [(0.02, 0.03), (0.03, 0.06), (0.07, 0.05)], 
        ]
        S = [19, 38, 76]
    
    writer = SummaryWriter(log_dir=f"logs/results_test/")

    scaled_anchors = (
        torch.tensor(anchors) * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(device)
    
    loss_f = YoloV5Loss()

    for run in range(args.nruns):

        test_dataset = CoCoDataset(
            "data/coco/test_10examples.csv",
            "data/coco/images/",
            "data/coco/labels/",
            img_size=img_size,
            mode="test",
        )
        
        test_loader = DataLoader(
            dataset=test_dataset,
            num_workers=args.nworkers,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )
        
        model = YoloV5_EfficientNet(nclasses=args.nclasses, gate=args.gate).to(device)
        # SSM has a lot it scalar operatins like .item() torch compile breaks them
        # so if we use ssm we need to capture sclar outputs 
        #torch._dynamo.config.capture_scalar_outputs = True
        #model = torch.compile(model)
    
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.momentum, 0.999),
        )
        #ema = EMA(model, decay=args.ema_decay, device=device)
        # Use constant learning rate instead of OneCycleLR
        scheduler = optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,  # Keep learning rate constant
            total_iters=args.nepochs
        )
        scaler = GradScaler()
        
        # epoch progress bar
        epoch_pbar = tqdm(range(args.nepochs), desc=f"Run {run+1}/{args.nruns}", position=0)
        
        for epoch in epoch_pbar:
            # Training phase
            model.train()

            train_loss, train_class_acc, train_noobj_acc, train_obj_acc = 0, 0, 0, 0
            
            # training progress bar
            train_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{args.nepochs} Training", leave=False, position=1)
            
            for batch_idx, (x, y) in enumerate(train_pbar):
                x = x.permute(0, 3, 1, 2).to(device)
                y0, y1, y2 = y[0].to(device), y[1].to(device), y[2].to(device)
                
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast():
                    preds = model(x)
                    loss = (loss_f(preds[0], y0, scaled_anchors[0]) + 
                        loss_f(preds[1], y1, scaled_anchors[1]) + 
                        loss_f(preds[2], y2, scaled_anchors[2]))
                    
                # Store the original loss for metrics
                original_loss = loss.item()
                    
                scaler.scale(loss).backward()
                
                # unscale bcs of mixed preicison
                scaler.unscale_(optimizer)
                # clip gradients since EMA + mixed precison can cause instability
                # YoloV4 also uses SiLU known for instability
                #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                
                scaler.step(optimizer)
                scaler.update()
                #ema.update(model)
                scheduler.step()
                
                # Update metrics
                train_loss += original_loss
                class_acc, noobj_acc, obj_acc = class_accuracy(rank=device,
                    model_pred=preds, 
                    target=y, 
                    confidence_threshold=0.001
                )
                train_class_acc += class_acc
                train_noobj_acc += noobj_acc
                train_obj_acc += obj_acc
                
                # Update training progress bar with current loss
                train_pbar.set_postfix({
                    'Loss': f'{original_loss:.4f}'
                })
            
            # Average metrics over batches
            train_loss /= len(test_loader)
            train_class_acc /= len(test_loader)
            train_noobj_acc /= len(test_loader)
            train_obj_acc /= len(test_loader)
            
            # Testing phase
            model.eval()
            test_loss, test_class_acc, test_noobj_acc, test_obj_acc = 0, 0, 0, 0
            
            # Apply EMA parameters for testing
            #ema.apply_shadow()
            
            # testing progress bar
            test_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{args.nepochs} Testing", leave=False, position=1)
            
            with torch.no_grad():
                for batch_idx, (x, y) in enumerate(test_pbar):
                    x = x.permute(0, 3, 1, 2).to(device)
                    y0, y1, y2 = y[0].to(device), y[1].to(device), y[2].to(device)
                    
                    with torch.cuda.amp.autocast():
                        preds = model(x)
                        loss = (loss_f(preds[0], y0, scaled_anchors[0]) + 
                            loss_f(preds[1], y1, scaled_anchors[1]) + 
                            loss_f(preds[2], y2, scaled_anchors[2]))
                    
                    test_loss += loss.item()
                    class_acc, noobj_acc, obj_acc = class_accuracy(rank=device,
                        model_pred=preds, 
                        target=y, 
                        confidence_threshold=0.001
                    )
                    test_class_acc += class_acc
                    test_noobj_acc += noobj_acc
                    test_obj_acc += obj_acc
                    
                    # update testing progress bar
                    test_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            # Restore original parameters after testing
            #ema.restore()
            
            test_loss /= len(test_loader)
            test_class_acc /= len(test_loader)
            test_noobj_acc /= len(test_loader)
            test_obj_acc /= len(test_loader)
            
            # Compute metrics periodically
            if (epoch + 1 == 1) or ((epoch + 1) % 10 == 0) or (epoch + 1 >= args.nepochs - 1):
                # Apply EMA parameters for metrics
                #ema.apply_shadow()
                
                test_pred_boxes, test_true_boxes = get_bounding_boxes(device,
                    test_loader,
                    model,
                    nms_iou_threshold=0.6,
                    confidence_threshold=0.001,
                    anchors=anchors,
                    use_torch_nms=True
                )
                
                # Calculate mAP
                test_map, test_recall, test_precision = mean_average_precision(device,
                    test_pred_boxes,
                    test_true_boxes,
                    iou_threshold=0.5,
                    nclasses=args.nclasses
                )
                
                # Restore original parameters after metrics
                #ema.restore()
                
                # Update epoch progress bar with key metrics
                epoch_pbar.set_postfix({
                    'Train Loss': f'{train_loss:.4f}',
                    'Test mAP': f'{test_map:.4f}',
                })
                
                print(f"Run:{run+1}/{args.nruns} Epoch:{epoch + 1}/{args.nepochs}")
                print(f"Train[Loss:{train_loss:.4f} Class Acc:{train_class_acc:.4f} NoObj acc:{train_noobj_acc:.4f} Obj Acc:{train_obj_acc:.4f}]")
                print(f"Test[Loss:{test_loss:.4f} Class Acc:{test_class_acc:.4f} NoObj Acc:{test_noobj_acc:.4f} Obj Acc:{test_obj_acc:.4f}]")
                print(f"Metrics[mAP@0.5:{test_map:.4f} Precision:{test_precision:.4f} Recall:{test_recall:.4f}]")
                print(f"Detections[Predicted:{len(test_pred_boxes)} True:{len(test_true_boxes)}]")
                
                # TensorBoard logging
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/test', test_loss, epoch)
                writer.add_scalar('Accuracy/class', train_class_acc, epoch)
                writer.add_scalar('Accuracy/noobj', train_noobj_acc, epoch)
                writer.add_scalar('Accuracy/obj', train_obj_acc, epoch)
                writer.add_scalar('mAP/test', test_map, epoch)
                writer.add_scalar('Precision/test', test_precision, epoch)
                writer.add_scalar('Recall/test', test_recall, epoch)

        epoch_pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nruns", type=int, default=1, help="Number of runs to perform")
    parser.add_argument("--nepochs", type=int, default=100, help="Number of epochs to perform")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--momentum", type=float, default=0.937, help="Momentum for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0000, help="Weight decay for optimizer")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay rate")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--nclasses", type=int, default=80, help="Number of classes")
    parser.add_argument("--nworkers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--gate", type=str, default=None, help="Recurrence for global context.")
    parser.add_argument("--target_batch_size", type=int, default=32, help="Target batchsize for gradient accumulation")

    args = parser.parse_args()
    
    print("Training configuration:")
    print(f"- Runs: {args.nruns}")
    print(f"- Epochs: {args.nepochs}")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Learning rate: {args.lr}")
    print(f"- Momentum: {args.momentum}")
    print(f"- Weight decay: {args.weight_decay}")
    print(f"- Workers: {args.nworkers}")
    
    main(args)