import torch
from utils.utils import class_accuracy_enas
from torch.cuda.amp import autocast
import torch.distributed as dist

def trainholov4_enas_vid_bptt(
    rank,
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

    model_yolo.train()
    model_rnn.train()
    # calculate the accumulation steps needed to simulate the target batch size
    actual_batch_size = train_loader.batch_size
    accumulation_steps = target_batch_size // (actual_batch_size * ngpus)
    for batch_idx, (x, y) in enumerate(train_loader):
        #print(f"Iteration: {batch_idx}/{len(train_loader)}")
        loss_lst = []
        class_acc_lst, noobj_acc_lst, obj_acc_lst = [], [], []
        # iterate over sequence
        last_ts = ((0), (0), (0), (0))
        for j in range(len(x)):
            #print(f"Seq:{j}/{len(x)}")
            x_t = x[j].permute(0, 3, 1, 2)
            x_t = x_t.to(rank)
            y0 = y[j][0].to(rank)
            # x shape :-: (batchsize, channels, height, width)
            with autocast():
                preds = model_yolo(x=x_t)
                yolo_loss = (
                    loss_yolo_f(preds, y0, scaled_anchors[0], mode=mode)
                    )       

                class_acc, noobj_acc, obj_acc = class_accuracy_enas(rank, preds, y[j], conf_thresh)
                loss_lst.append(yolo_loss)
                class_acc_lst.append(class_acc)
                noobj_acc_lst.append(noobj_acc)
                obj_acc_lst.append(obj_acc)

                preds = model_rnn(preds)

                rnn_loss = loss_rnn_f(preds, y0, scaled_anchors[0], mode=mode)
                scaler.scale(rnn_loss).backward(retain_graph=True)
                optimizer_rnn.step()
                optimizer_rnn.zero_grad()
                scheduler_rnn.step()

        seq_loss = sum(loss_lst) / len(loss_lst)
        seq_class_acc = sum(class_acc_lst) / len(class_acc_lst)
        seq_noobj_acc = sum(noobj_acc_lst) / len(noobj_acc_lst)
        seq_obj_acc = sum(obj_acc_lst) / len(obj_acc_lst)

        # Normalize loss for accumulation
        scaler.scale(seq_loss / accumulation_steps).backward()
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1 == len(train_loader)):
            scaler.step(optimizer_yolo)
            scaler.update()
            optimizer_yolo.zero_grad()
            scheduler_yolo.step()

    return (seq_loss, seq_class_acc, seq_noobj_acc, seq_obj_acc)

def testholov4_enas_vid(
    rank,
    val_loader,
    model_yolo,
    model_rnn,
    loss_yolo_f,
    loss_rnn_f,
    scaled_anchors,
    conf_thresh,
    mode="iou",
):
    """
    Evaluate HOLOv4 Enas model on video data over a sequence of frames. Each sequence is evaluated
    together using the mean loss over the sequence.

    Args:
        rank (int): Rank of the current process in distributed evaluation.
        val_loader (DataLoader): DataLoader containing validation data.
        model_yolo (nn.Module): YOLOv4 model.
        model_rnn (nn.Module): Recurrent neural network model.
        loss_yolo_f (callable): Loss function for the YOLOv4 model.
        loss_rnn_f (callable): Loss function for the RNN model.
        scaled_anchors (list): Anchors for the YOLOv4 model.
        conf_thresh (float): Confidence threshold for classification accuracy.
        mode (str): Mode for calculating loss. Default is "iou".

    Returns:
        tuple: Tuple containing aggregated loss and accuracy metrics for the entire batch.
    """

    model_yolo.eval()
    model_rnn.eval()

    for batch_idx, (x, y) in enumerate(val_loader):
        loss_lst = []
        class_acc_lst, noobj_acc_lst, obj_acc_lst = [], [], []
        # iterate over sequence
        for j in range(len(x)):
            x_t = x[j].permute(0, 3, 1, 2)
            x_t = x_t.to(rank)
            y0 = y[j][0].to(rank)

            with torch.no_grad():
                preds = model_yolo(x=x_t)
                yolo_loss = loss_yolo_f(preds, y0, scaled_anchors[0], mode=mode)

                class_acc, noobj_acc, obj_acc = class_accuracy_enas(rank, preds, y[j], conf_thresh)
                loss_lst.append(yolo_loss)
                class_acc_lst.append(class_acc)
                noobj_acc_lst.append(noobj_acc)
                obj_acc_lst.append(obj_acc)

                preds = model_rnn(preds)
                rnn_loss = loss_rnn_f(preds, y0, scaled_anchors[0], mode=mode)

        seq_loss = sum(loss_lst) / len(loss_lst)
        seq_class_acc = sum(class_acc_lst) / len(class_acc_lst)
        seq_noobj_acc = sum(noobj_acc_lst) / len(noobj_acc_lst)
        seq_obj_acc = sum(obj_acc_lst) / len(obj_acc_lst)

    return (seq_loss, seq_class_acc, seq_noobj_acc, seq_obj_acc)