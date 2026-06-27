import torch
from utils.utils import top1accuracy, top5accuracy
import torch.nn.functional as F

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

def trainhippo(arguments, train_loader, model, optimizer, loss_f):
    """
    Input: train loader (torch loader), model (torch model), optimizer (torch optimizer)
          loss function (torch custom yolov1 loss).
    Output: loss (torch float).
    """
    model.train()
    loss_lst = []
    top1_acc_lst = []
    top5_acc_lst = [] 
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        # turn [64, 28, 28] to [64, 784]
        x = x.view(x.shape[0], -1).to(device)
        # turn [64, 784] to [784, 64, 1, 1]
        x = x.T.unsqueeze(-1).unsqueeze(-1).to(device)
        out, rec = model(x)
        del x, rec
        loss_val = loss_f(out, y)
        class_prob = F.softmax(out, dim = 1)
        del out
        # preds = class_prob.argmax(dim=1)
        loss_lst.append(float(loss_val.item()))
        top1_acc_val = top1accuracy(class_prob, y)
        top5_acc_val = top5accuracy(class_prob, y)
        top1_acc_lst.append(float(top1_acc_val))
        top5_acc_lst.append(float(top5_acc_val))
        del y, class_prob
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
    # compute average loss
    loss_val = round(sum(loss_lst) / len(loss_lst), 4)
    top1_acc = round(sum(top1_acc_lst) / len(top1_acc_lst),  4)
    top5_acc = round(sum(top5_acc_lst) / len(top5_acc_lst), 4)
   
    return (loss_val, top1_acc, top5_acc) 

def evaluatehippo(arguments, data_loader, model, loss_f):
    """
    Input: train loader (torch loader), model (torch model), optimizer (torch optimizer)
          loss function (torch custom yolov1 loss).
    Output: loss (torch float).
    """
    loss_lst = []
    top1_acc_lst = []
    top5_acc_lst = []
    model.eval()
    for batch_idx, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)
        x = x.view(x.shape[0], -1).to(device)
        x = x.T.unsqueeze(-1).unsqueeze(-1).to(device)
        out, rec = model(x)
        del x, rec
        class_prob = F.softmax(out, dim = 1)
        # pred = torch.argmax(class_prob, dim = 1)
        loss_val = loss_f(out, y)
        del out
        loss_lst.append(float(loss_val.item()))
        top1_acc_val = top1accuracy(class_prob, y)
        top5_acc_val = top5accuracy(class_prob, y)
        del y, class_prob
        top1_acc_lst.append(float(top1_acc_val))
        top5_acc_lst.append(float(top5_acc_val))

    # compute average loss
    loss_val = round(sum(loss_lst) / len(loss_lst), 4)
    top1_acc = round(sum(top1_acc_lst) / len(top1_acc_lst),  4)
    top5_acc = round(sum(top5_acc_lst) / len(top5_acc_lst), 4)
    return (loss_val, top1_acc, top5_acc)