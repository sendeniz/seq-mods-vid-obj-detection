import torch
from utils.utils import top1accuracy, top5accuracy
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(arguments, train_loader, model, optimizer, loss_f):
    """
    Performs the training loop.
    Input: train loader (torch loader)
           model (torch model)
           optimizer (torch optimizer)
           loss function (torch loss).
    Output: No output.
    """
    model.train()
    loss_lst = []
    top1_acc_lst = []
    top5_acc_lst = []
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        #print("y.shape:", y.shape)
        # turn [64, 28, 28] to [64, 784, 1]
        x = x.view(x.shape[0], -1, arguments["input_size"])
        # hippo rnn returns a tensor of shape (batchsize, 1, output_size)
        # the extra dimension at the first index is removed with if statement below.
        out = model(x)
        #print("out.shape:",out.shape)
        # if statement to ensure that out shape [batchsize, 1, outputsize]
        if out.shape == torch.Size([x.shape[0], 1, arguments["output_size"]]):
            out = out.squeeze(1)

        loss_val = loss_f(out, y)
        class_prob = F.softmax(out, dim=1)
        # preds = class_prob.argmax(dim=1)
        loss_lst.append(float(loss_val.item()))
        top1_acc_val = top1accuracy(class_prob, y)
        top5_acc_val = top5accuracy(class_prob, y)
        top1_acc_lst.append(float(top1_acc_val))
        top5_acc_lst.append(float(top5_acc_val))

        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
    # compute average loss
    loss_val = round(sum(loss_lst) / len(loss_lst), 4)
    top1_acc = round(sum(top1_acc_lst) / len(top1_acc_lst), 4)
    top5_acc = round(sum(top5_acc_lst) / len(top5_acc_lst), 4)

    return (loss_val, top1_acc, top5_acc)


def evaluate(arguments, data_loader, model, loss_f):
    """
    Input: test or train loader (torch loader)
           model (torch model)
           loss function (torch loss)
    Output: loss (torch float)
            accuracy (torch float)
    """
    loss_lst = []
    top1_acc_lst = []
    top5_acc_lst = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            x = x.view(x.shape[0], -1, arguments["input_size"])
            out = model(x)
            if out.shape == torch.Size([x.shape[0], 1, arguments["output_size"]]):
                out = out.squeeze(1)
            loss_val = loss_f(out, y)
            class_prob = F.softmax(out, dim=1)
            # pred = torch.argmax(class_prob, dim = 1)
            loss_lst.append(float(loss_val.item()))
            top1_acc_val = top1accuracy(class_prob, y)
            top5_acc_val = top5accuracy(class_prob, y)
            top1_acc_lst.append(float(top1_acc_val))
            top5_acc_lst.append(float(top5_acc_val))

        # compute average loss
        loss_val = round(sum(loss_lst) / len(loss_lst), 4)
        top1_acc = round(sum(top1_acc_lst) / len(top1_acc_lst), 4)
        top5_acc = round(sum(top5_acc_lst) / len(top5_acc_lst), 4)
        return (loss_val, top1_acc, top5_acc)
