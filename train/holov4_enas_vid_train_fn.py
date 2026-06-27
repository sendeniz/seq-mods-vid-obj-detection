import torch
from utils.utils import class_accuracy_enas
from torch.cuda.amp import autocast
import torch.distributed as dist

def trainholov4_enas_vid_bptt(
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
    target_batch_size = 128,
    ngpus = 1,
):
    """
    Train HOLOv4 Enas model on video data over a sequence of frames. Each sequence is optimized
    together using the mean loss over the sequence.

    Args:
        rank (int): Rank of the current process in distributed training.
        train_loader (DataLoader): DataLoader containing training data.
        model (nn.Module): YOLOv4 model.
        optimizer (Optimizer): Optimizer for updating model parameters.
        scheduler (LRScheduler): Learning rate scheduler.
        loss_f (callable): Loss function.
        scaled_anchors (list): Anchors for the YOLOv4 model.
        scaler (GradScaler): Gradient scaler for mixed precision training.
        conf_thresh (float): Confidence threshold for classification accuracy.
        mode (str): Mode for calculating loss. Default is "iou".

    Returns:
        tuple: Tuple containing aggregated loss and accuracy metrics for the entire batch.
    """

    model.train()
    # calculate the accumulation steps needed to simulate the target batch size
    actual_batch_size = train_loader.batch_size
    accumulation_steps = target_batch_size // (actual_batch_size * ngpus)
    for batch_idx, (x, y) in enumerate(train_loader):
        loss_lst = []
        class_acc_lst, noobj_acc_lst, obj_acc_lst = [], [], []
        # iterate over sequence
        carry = ((None, None), (None, None), (None, None), (None, None))
        for j in range(len(x)):
            x_t = x[j].permute(0, 3, 1, 2)
            x_t = x_t.to(rank)
            y0 = y[j][0].to(rank)
            # x shape :-: (batchsize, channels, height, width)
            with autocast():
                preds, carry = model(x=x_t, t=j, carry=carry)
                loss = (
                    loss_f(preds, y0, scaled_anchors[0], mode=mode)
                    )       

                class_acc, noobj_acc, obj_acc = class_accuracy_enas(rank, preds, y[j], conf_thresh)
                loss_lst.append(loss)
                class_acc_lst.append(class_acc)
                noobj_acc_lst.append(noobj_acc)
                obj_acc_lst.append(obj_acc)

        seq_loss = sum(loss_lst) / len(loss_lst)
        seq_class_acc = sum(class_acc_lst) / len(class_acc_lst)
        seq_noobj_acc = sum(noobj_acc_lst) / len(noobj_acc_lst)
        seq_obj_acc = sum(obj_acc_lst) / len(obj_acc_lst)

        # Normalize loss for accumulation
        scaler.scale(seq_loss / accumulation_steps).backward()
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1 == len(train_loader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

    return (seq_loss, seq_class_acc, seq_noobj_acc, seq_obj_acc)

def testholov4_enas_vid(rank,
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
            loss_lst = []
            class_acc_lst, noobj_acc_lst, obj_acc_lst = [], [], []
            # iterate over sequence
            carry = ((None, None), (None, None), (None, None), (None, None))
            for j in range(len(x)):
                x_t = x[j].permute(0, 3, 1, 2)
                x_t = x_t.to(rank)
                y0 = y[j][0].to(rank)
                with autocast():
                    preds, carry = model(x=x_t, t=j, carry=carry)

                    loss = (
                        loss_f(preds, y0, scaled_anchors[0], mode=mode)
                        )
                    
                class_acc, noobj_acc, obj_acc = class_accuracy_enas(rank, preds, y[j], conf_thresh)
                loss_lst.append(loss)
                class_acc_lst.append(class_acc)
                noobj_acc_lst.append(noobj_acc)
                obj_acc_lst.append(obj_acc)

            seq_loss = sum(loss_lst) / len(loss_lst)
            seq_class_acc = sum(class_acc_lst) / len(class_acc_lst)
            seq_noobj_acc = sum(noobj_acc_lst) / len(noobj_acc_lst)
            seq_obj_acc = sum(obj_acc_lst) / len(obj_acc_lst)
    return (seq_loss, seq_class_acc, seq_noobj_acc, seq_obj_acc)


def trainholov4_enas_vid_no_bptt(
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
    target_batch_size = 128,
    ngpus = 1,
):
    """
    Train HOLOv4 Enas model on video data over a sequence of frames. Each sequence is optimized
    individually.

    Args:
        rank (int): Rank of the current process in distributed training.
        train_loader (DataLoader): DataLoader containing training data.
        model (nn.Module): YOLOv4 model.
        optimizer (Optimizer): Optimizer for updating model parameters.
        scheduler (LRScheduler): Learning rate scheduler.
        loss_f (callable): Loss function.
        scaled_anchors (list): Anchors for the YOLOv4 model.
        scaler (GradScaler): Gradient scaler for mixed precision training.
        conf_thresh (float): Confidence threshold for classification accuracy.
        mode (str): Mode for calculating loss. Default is "iou".

    Returns:
        tuple: Tuple containing aggregated loss and accuracy metrics for the entire batch.
    """

    model.train()
    actual_batch_size = train_loader.batch_size
    accumulation_steps = target_batch_size // (actual_batch_size * ngpus)

    for batch_idx, (x, y) in enumerate(train_loader):
        #print(f"Iteration: {batch_idx}/{len(train_loader)}")
        class_acc_lst, noobj_acc_lst, obj_acc_lst = [], [], []
        carry = ((None, None), (None, None), (None, None), (None, None))        
        # Iterate over each frame in the sequence
        for j in range(len(x)):
            # Prepare the input and target for the current frame
            x_t = x[j].permute(0, 3, 1, 2).to(rank)
            y0 = y[j][0].to(rank)

            with autocast():
                # Forward pass for the current frame
                preds, carry = model(x=x_t, t=j, carry=carry)

                # Compute loss for the current frame
                loss = loss_f(preds, y0, scaled_anchors[0], mode=mode)

                # Calculate accuracies for the current frame
                class_acc, noobj_acc, obj_acc = class_accuracy_enas(rank, preds, y[j], conf_thresh)
                class_acc_lst.append(class_acc)
                noobj_acc_lst.append(noobj_acc)
                obj_acc_lst.append(obj_acc)

                # Backward pass for the current frame 
                #scaler.scale(loss).backward()
                scaler.scale(loss).backward(retain_graph=True)

                # Optimizer step and gradient update
                if (batch_idx * len(x) + j + 1) % accumulation_steps == 0 or (batch_idx == len(train_loader) - 1 and j == len(x) - 1):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()

        # Calculate mean accuracies for the sequence
        seq_class_acc = sum(class_acc_lst) / len(class_acc_lst)
        seq_noobj_acc = sum(noobj_acc_lst) / len(noobj_acc_lst)
        seq_obj_acc = sum(obj_acc_lst) / len(obj_acc_lst)

        # Returning the last frame's loss and average accuracies for monitoring
        return (loss.item(), seq_class_acc, seq_noobj_acc, seq_obj_acc)