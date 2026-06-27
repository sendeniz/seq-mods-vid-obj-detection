import torch
from utils.utils import class_accuracy
from torch.cuda.amp import autocast
import torch.distributed as dist

def trainyolov4(
    rank,
    train_loader,
    model,
    optimizer,
    scheduler,
    loss_f,
    scaled_anchors,
    scaler,
    conf_thresh,
    mode="iou",
):
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.permute(0, 3, 1, 2)
        x = x.to(rank)
        y0, y1, y2 = (y[0].to(rank), y[1].to(rank), y[2].to(rank))
        # x shape :-: (batchsize, channels, height, width)
        with autocast():
            preds = model(x)
            loss = (
                loss_f(preds[0], y0, scaled_anchors[0], mode=mode)
                + loss_f(preds[1], y1, scaled_anchors[1], mode=mode)
                + loss_f(preds[2], y2, scaled_anchors[2], mode=mode)
            )
        class_acc, noobj_acc, obj_acc = class_accuracy(rank, preds, y, conf_thresh)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
    return (loss, class_acc, noobj_acc, obj_acc)

def testyolov4(rank,
    test_loader,
    model,
    loss_f,
    scaled_anchors,
    conf_thresh,
    mode="iou",
):
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x = x.permute(0, 3, 1, 2)
            x = x.to(rank)
            y0, y1, y2 = (y[0].to(rank), y[1].to(rank), y[2].to(rank))
            with autocast():
                preds = model(x)
                loss= (
                    loss_f(preds[0], y0, scaled_anchors[0], mode=mode)
                    + loss_f(preds[1], y1, scaled_anchors[1], mode=mode)
                    + loss_f(preds[2], y2, scaled_anchors[2], mode=mode)
                )
            class_acc, noobj_acc, obj_acc = class_accuracy(rank, preds, y, conf_thresh)
    return (loss, class_acc, noobj_acc, obj_acc)
