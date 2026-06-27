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
from loss.yolov4loss import YoloV4Loss2, YoloV4Loss3
from loss.yolov5loss import YoloV5Loss 
from utils.dataset import CoCoDataset
from utils.utils import mean_average_precision, get_bounding_boxes, class_accuracy
from utils.utils import mean_average_precision_fast
from utils.ema import EMA
from torch.cuda.amp import GradScaler
import os
from torch.utils.tensorboard import SummaryWriter
import argparse
import random
from tqdm import tqdm

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # if using multi-GPU.
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def worker_init_fn(worker_id):
    """Ensure each DataLoader worker gets a proper seed"""
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_size = 640 #608
    path_cpt_file = "ckpts/"
    
    # compute practical IoU thresholds
    # Focus on the most relevant thresholds
    # doing it over 0.5 to 0.95 in steps of 0.05 is expensive
    map_vals = [0.5, 0.75]  
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            best_map = checkpoint.get('mAP', 0)
            start_run = checkpoint.get('run', 0)
            print(f"Resuming from run {start_run}, epoch {start_epoch}")
        else:
            print(f"No checkpoint found at '{args.resume}'")
            best_map = 0
            start_epoch = 0
            start_run = 0
    else:
        best_map = 0
        start_epoch = 0
        start_run = 0
    
    if args.gate == None:
        model_name = 'yolov5'
    else:
        model_name = f'yolov5_{args.gate}'
    
    # Create directory if it doesn't exist
    os.makedirs(path_cpt_file, exist_ok=True) 
    
    # imgsize 640 YOlOV5
    # Large objects (assigned to smallest grid, e.g., 20x20)
    anchors = [
        [(0.22, 0.17), (0.30, 0.38), (0.72, 0.63)],
        # Medium objects (e.g., 40x40)
        [(0.06, 0.12), (0.12, 0.09), (0.11, 0.23)],
        # Small objects (e.g., 80x80)
        [(0.02, 0.03), (0.03, 0.06), (0.06, 0.04)],
    ]
    # Grid sizes (640/32=20, 640/16=40, 640/8=80)
    S = [20, 40, 80]  
    
    writer = SummaryWriter(log_dir=f"logs/results_{model_name}/")

    scaled_anchors = (
        torch.tensor(anchors) * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(device)
    
    loss_f = YoloV5Loss()

    #for run in range(args.nruns):
    for run in range(start_run, args.nruns):
        # Set different seed for each run for variability between runs
        set_seed(42 + run)
        
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
        
        train_loader = DataLoader(
            dataset=train_dataset,
            num_workers=args.nworkers,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )

        test_loader = DataLoader(
            dataset=test_dataset,
            num_workers=args.nworkers,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            worker_init_fn=worker_init_fn
        )
        
        model = YoloV5_EfficientNet(nclasses=args.nclasses, gate=args.gate).to(device)
        # SSM has a lot it scalar operatins like .item() torch compile breaks them
        # so if we use ssm we need to capture sclar outputs 
        torch._dynamo.config.capture_scalar_outputs = True
        model = torch.compile(model)
        accumulation_steps = args.target_batch_size // (args.batch_size)
    
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.momentum, 0.999),
        )
        ema = EMA(model, decay=args.ema_decay, device=device)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=args.nepochs,
            steps_per_epoch=(
                len(train_loader) // (args.target_batch_size // (args.batch_size))
                ) + 1,
            anneal_strategy="cos",
        )
        scaler = GradScaler()
        
        
        # Load state dicts from checkpoint
        if args.resume:
            if os.path.isfile(args.resume):
                print(f"Loading state dicts '{args.resume}'")
                # Load model state
                model.load_state_dict(checkpoint['model_state_dict'])
                # Load optimizer state
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # Load scheduler state
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                # Load scaler state if available
                if 'scaler_state_dict' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler_state_dict'])
                else:
                     print("No scaler_state_dict found, starting new GradScaler.")
                if 'ema_state_dict' in checkpoint:
                    ema.load_state_dict(checkpoint['ema_state_dict'])
                    print("Loaded EMA state from checkpoint")
        
        # epoch progress bar
        epoch_pbar = tqdm(range(start_epoch, args.nepochs), desc=f"Run {run+1}/{args.nruns}", position=0)
        
        for epoch in epoch_pbar:
            # Training phase
            model.train()

            train_loss, train_class_acc, train_noobj_acc, train_obj_acc = 0, 0, 0, 0
            
            optimizer.zero_grad()  
            
            # training progress bar
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.nepochs} Training", leave=False, position=1)
            
            for batch_idx, (x, y) in enumerate(train_pbar):
                x = x.permute(0, 3, 1, 2).to(device)
                y0, y1, y2 = y[0].to(device), y[1].to(device), y[2].to(device)
                
                with torch.cuda.amp.autocast():
                    preds = model(x)
                    loss = (loss_f(preds[0], y0, scaled_anchors[0]) + 
                        loss_f(preds[1], y1, scaled_anchors[1]) + 
                        loss_f(preds[2], y2, scaled_anchors[2]))
                    
                # Store the original loss for metrics
                original_loss = loss.item()
                    
                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    
                    # unscale bcs of mixed preicison
                    scaler.unscale_(optimizer)
                    # clip gradients since EMA + mixed precison can cause instability
                    # YoloV4 also uses SiLU known for instability
                    #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    
                    scaler.step(optimizer)
                    scaler.update()
                    ema.update(model)
                    optimizer.zero_grad()
                    scheduler.step()
                
                # Update metrics
                train_loss += original_loss
                class_acc, noobj_acc, obj_acc = class_accuracy(rank=device,
                    model_pred=preds, 
                    target=y, 
                    confidence_threshold=0.8
                )
                train_class_acc += class_acc
                train_noobj_acc += noobj_acc
                train_obj_acc += obj_acc
                
                # Update training progress bar with current loss
                train_pbar.set_postfix({
                    'Loss': f'{original_loss:.4f}',
                    'Accum Step': f'{(batch_idx + 1) % accumulation_steps}/{accumulation_steps}'
                })
            
            # Average metrics over batches
            train_loss /= len(train_loader)
            train_class_acc /= len(train_loader)
            train_noobj_acc /= len(train_loader)
            train_obj_acc /= len(train_loader)
            
            # Testing phase
            model.eval()
            test_loss, test_class_acc, test_noobj_acc, test_obj_acc = 0, 0, 0, 0
            
            # Apply EMA parameters for testing
            ema.apply_shadow()
            
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
                        confidence_threshold=0.8
                    )
                    test_class_acc += class_acc
                    test_noobj_acc += noobj_acc
                    test_obj_acc += obj_acc
                    
                    # update testing progress bar
                    test_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            # Restore original parameters after testing
            ema.restore()
            
            test_loss /= len(test_loader)
            test_class_acc /= len(test_loader)
            test_noobj_acc /= len(test_loader)
            test_obj_acc /= len(test_loader)
            
            # Compute metrics periodically
            if (epoch + 1 == 1) or ((epoch + 1) % 10 == 0) or (epoch + 1 >= args.nepochs - 1):
                # Progress bar for metric computation
                metric_pbar = tqdm(total=len(map_vals), desc="Computing mAP", leave=False, position=1)
                
                # Apply EMA parameters for metrics
                ema.apply_shadow()
                
                test_pred_boxes, test_true_boxes = get_bounding_boxes(device,
                    test_loader,
                    model,
                    nms_iou_threshold=0.6,
                    confidence_threshold=0.8,
                    anchors=anchors
                )
                
                # Only compute key mAP values to save compute
                test_map_05, test_recall_05, test_precision_05 = mean_average_precision_fast(device,
                test_pred_boxes,
                test_true_boxes,
                iou_threshold=0.5,
                nclasses=args.nclasses
                )

                # Around line 283:
                test_map_075, test_recall_075, test_precision_075 = mean_average_precision_fast(device,
                    test_pred_boxes,
                    test_true_boxes,
                    iou_threshold=0.75,
                    nclasses=args.nclasses
                )
                metric_pbar.update(1)
                
                # Calculate mean mAP across key thresholds
                test_map = (test_map_05 + test_map_075) / 2
                test_recall = (test_recall_05 + test_recall_075) / 2
                test_precision = (test_precision_05 + test_precision_075) / 2
                
                metric_pbar.close()
                
                # Restore original parameters after metrics
                ema.restore()
                
                # Update epoch progress bar with key metrics
                epoch_pbar.set_postfix({
                    'Train Loss': f'{train_loss:.4f}',
                    'Test mAP': f'{test_map:.4f}',
                    'Best mAP': f'{best_map:.4f}'
                })
                
                print(f"Run:{run+1}/{args.nruns} Epoch:{epoch + 1}/{args.nepochs}")
                print(f"Train[Loss:{train_loss:.4f} Class Acc:{train_class_acc:.4f} NoObj acc:{train_noobj_acc:.4f} Obj Acc:{train_obj_acc:.4f}]")
                print(f"Test[Loss:{test_loss:.4f} Class Acc:{test_class_acc:.4f} NoObj Acc:{test_noobj_acc:.4f} Obj Acc:{test_obj_acc:.4f}]")
                print(f"Metrics[mAP@0.5:{test_map_05:.4f} Precision:{test_precision_05:.4f} Recall:{test_recall_05:.4f}]")
                print(f"Metrics[mAP@0.75:{test_map_075:.4f} Precision:{test_precision_075:.4f} Recall:{test_recall_075:.4f}]")
                print(f"Metrics[mAP@0.5:0.75:{test_map:.4f} Precision:{test_precision:.4f} Recall:{test_recall:.4f}]")
                print(f"Detections[Predicted:{len(test_pred_boxes)} True:{len(test_true_boxes)}]")
                
                # Save checkpoint
                if args.save_model:
                    checkpoint = {
                        'run': run,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'ema_state_dict': ema.state_dict(),
                        'loss': train_loss,
                        'mAP': test_map,
                        'args': vars(args),  # Save training arguments
                        'best_map': best_map,
                    }
                    
                    torch.save(checkpoint, path_cpt_file + f"{model_name}_run_{run}_latest.pth")
                    
                    # Store best map checkpoint if we find it
                    if test_map > best_map:
                        best_map = test_map
                        checkpoint['best_map'] = best_map
                        torch.save(checkpoint, path_cpt_file + f"{model_name}_run_{run}_best_map.pth")
                        print(f"New best mAP: {best_map:.4f}, saved best model")
                    
                    # TensorBoard logging
                    writer.add_scalar('Loss/train', train_loss, epoch)
                    writer.add_scalar('Loss/test', test_loss, epoch)
                    writer.add_scalar('Accuracy/class', train_class_acc, epoch)
                    writer.add_scalar('Accuracy/noobj', train_noobj_acc, epoch)
                    writer.add_scalar('Accuracy/obj', train_obj_acc, epoch)
                    writer.add_scalar('mAP/test', test_map, epoch)
                    writer.add_scalar('Precision/test', test_precision, epoch)
                    writer.add_scalar('Recall/test', test_recall, epoch)
                    writer.add_scalar('mAP/mAP_0.5', test_map_05, epoch)
                    writer.add_scalar('mAP/mAP_0.75', test_map_075, epoch)

        epoch_pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nruns", type=int, default=1, help="Number of runs to perform")
    parser.add_argument("--nepochs", type=int, default=200, help="Number of epochs to perform")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--momentum", type=float, default=0.937, help="Momentum for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay for optimizer")
    parser.add_argument("--ema_decay", type=float, default=0.9999, help="EMA decay rate")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--nclasses", type=int, default=80, help="Number of classes")
    parser.add_argument("--nworkers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--save_model", type=bool, default=False, help="Store model checkpoints")
    parser.add_argument("--gate", type=str, default=None, help="Recurrence for global context.")
    parser.add_argument("--target_batch_size", type=int, default=128, help="Target batchsize for gradient accumulation")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")


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
    if args.resume:
        print(f"- Resuming from: {args.resume}")
    
    main(args)