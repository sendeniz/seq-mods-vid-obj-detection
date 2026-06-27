
      "current_model": "",
      "lr": 1e-4, # inital learning rate
      "warmup_lr": 1e-3, # learning rate used for linear warmup: cosine_anneal and exponential decay 
      "min_lr": 1e-5, 
      "cosine_anneal": False, # performs internal warmup so no linear warmup needed
      "exponential_decay": False,
      "onecycle": False,
      "path_cpt_file": ""
  }
  arguments = {**arguments, **_arguments}

  for key, value in arguments.items():
    if key in floatTypes:
        arguments[key] = float(value)
    if key in intTypes:
        arguments[key] = int(value)
    if key in boolTypes:
        if value == "False":
            arguments[key] = False
        if value == "True":
            arguments[key] = True
         
  if arguments["model_name"] not in model_names:
    print(f"model name {arguments['model_name']} was not found, use simple, gru, lstm, hippo, gatedhippo or yolov4_416 or yolov4_608.")
    return
  if arguments["dataset_name"] not in dataset_names:
    print(f"dataset name {arguments['dataset_name']} was not found")
    return
  if arguments["model_name"] == "rnn":
    arguments["init_rnn"] = True
    arguments["current_model"] = model_names[0]
    arguments["path_cpt_file"] = 'cpts/{}_mnist.cpt'.format(arguments["current_model"])
    arguments["model_name"] = 'Simple Rnn'
  elif arguments["model_name"] == "gru":
    arguments["init_grurnn"] = True
    arguments["current_model"] = model_names[1]
    arguments["path_cpt_file"] = 'cpts/{}_rnn_mnist.cpt'.format(arguments["current_model"])
    arguments["model_name"] = 'Gru Rnn'
  elif arguments["model_name"] == "lstm":
    arguments["init_lstmrnn"] = True
    arguments["current_model"] = model_names[2]
    arguments["path_cpt_file"] = 'cpts/{}_rnn_mnist.cpt'.format(arguments["current_model"])
    arguments["model_name"] = 'Lstm Rnn'
  elif arguments["model_name"] == "urlstm":
    arguments["init_urlstmrnn"] = True
    arguments["current_model"] = model_names[3]
    arguments["path_cpt_file"] = 'cpts/{}_rnn_mnist.cpt'.format(arguments["current_model"])
    arguments["model_name"] = 'Ur Lstm Rnn'
  elif arguments["model_name"] == "hippo":
    arguments["init_hippo"] = True
    arguments["current_model"] = model_names[4]
    arguments["path_cpt_file"] = 'cpts/{}_mnist.cpt'.format(arguments["current_model"])
    arguments["model_name"] = 'Simple Hippo'
  elif arguments["model_name"] == "gatedhippo":
    arguments["init_hippornn"] = True
    arguments["current_model"] = model_names[5]
    arguments["path_cpt_file"] = 'cpts/{}_rnn_mnist.cpt'.format(arguments["current_model"])
    arguments["model_name"] = 'Hippo Rnn'
  # for object detect
  elif arguments["model_name"] == "yolov4_608":
    arguments["init_yolov4"] = True
    arguments["current_model"] = model_names[6]
    arguments["path_cpt_file"] = 'cpts/{}_mscoco.cpt'.format(arguments["current_model"])
    arguments["model_name"] = 'YoloV4 608'
    # yolo anchors rescaled between 0,1
    # yolo scales and anchors for image size 608, 608
    S = [19, 38, 76]  
    anchors = [
       [(0.23, 0.18), (0.32, 0.40), (0.75, 0.66)],
       [(0.06, 0.12), (0.12, 0.09), (0.12, 0.24)],
       [(0.02, 0.03), (0.03, 0.06), (0.07, 0.05)],
       ]
 
  elif arguments["model_name"] == "yolov4_416":
    arguments["init_yolov4"] = True
    arguments["current_model"] = model_names[7]
    arguments["path_cpt_file"] = 'cpts/{}_mscoco.cpt'.format(arguments["current_model"])
    arguments["model_name"] = 'YoloV4 416'
    # yolo scales and anchors for image size 416, 416
    S = [13, 26, 52]
    anchors = [
       [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
       [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
       [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
       ]

  if arguments["dataset_name"] == "mnist":
      arguments["mnist"] = True
  if arguments["dataset_name"] == "mscoco":
      arguments["mscoco"] = True

  main(arguments)

def train(arguments, train_loader, model, optimizer, loss_f):
    '''
    Performs the training loop. 
    Input: train loader (torch loader)
           model (torch model)
           optimizer (torch optimizer)
           loss function (torch loss).
    Output: No output.
    '''
    model.train()
    loss_lst = []
    top1_acc_lst = []
    top5_acc_lst = [] 
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        # turn [64, 28, 28] to [64, 784, 1]
        x = x.view(x.shape[0], -1, arguments['input_size'])
        # hippo rnn returns a tensor of shape (batchsize, 1, output_size)
        # the extra dimension at the first index is removed with if statement below.
        out = model(x)
        if out.shape == torch.Size([arguments['batch_size'], 1, arguments['output_size']]):
            out = out.squeeze(1)
        loss_val = loss_f(out, y)
        class_prob = F.softmax(out, dim = 1)
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
    top1_acc = round(sum(top1_acc_lst) / len(top1_acc_lst),  4)
    top5_acc = round(sum(top5_acc_lst) / len(top5_acc_lst), 4)
   
    return (loss_val, top1_acc, top5_acc) 

def evaluate(arguments, data_loader, model, loss_f):
    '''
    Input: test or train loader (torch loader) 
           model (torch model)
           loss function (torch loss)
    Output: loss (torch float)
            accuracy (torch float)
    '''
    loss_lst = []
    top1_acc_lst = []
    top5_acc_lst = []
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            x = x.view(x.shape[0], -1, arguments['input_size'])
            out = model(x)
            if out.shape == torch.Size([arguments['batch_size'], 1, arguments['output_size']]):
                out =  out.squeeze(1)
            loss_val = loss_f(out, y)
            class_prob = F.softmax(out, dim = 1)
            # pred = torch.argmax(class_prob, dim = 1)
            loss_lst.append(float(loss_val.item()))
            top1_acc_val = top1accuracy(class_prob, y)
            top5_acc_val = top5accuracy(class_prob, y)
            top1_acc_lst.append(float(top1_acc_val))
            top5_acc_lst.append(float(top5_acc_val))

        # compute average loss
        loss_val = round(sum(loss_lst) / len(loss_lst), 4)
        top1_acc = round(sum(top1_acc_lst) / len(top1_acc_lst),  4)
        top5_acc = round(sum(top5_acc_lst) / len(top5_acc_lst), 4)
        return (loss_val, top1_acc, top5_acc)


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

def trainyolov4(arguments, train_loader, model, optimizer, scheduler, loss_f, scaled_anchors):
    model.train()
    # accumulate gradients over n accum_iters using multple batches
    # and update once minibatches are as large as a batchsize of 64
    accum_iter = 64 / arguments["batch_size"]
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.permute(0, 3, 1, 2)
        x = x.to(torch.float32).to(device)
        y0, y1, y2 = (y[0].to(device), y[1].to(device), y[2].to(device))
        # x shape :-: (batchsize, channels, height, width)
        preds = model(x)
        loss_val = (loss_f(preds[0], y0, scaled_anchors[0]) + loss_f(preds[1], y1, scaled_anchors[1]) + loss_f(preds[2], y2, scaled_anchors[2]))
        class_acc, noobj_acc, obj_acc = class_accuracy(preds, y, arguments["conf_thresh"])
        loss_val.backward()
        if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
    return (float(loss_val.item()), class_acc, noobj_acc, obj_acc)

def testyolov4(arguments, test_loader, model, loss_f, scaled_anchors):
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(test_loader):
            x = x.permute(0, 3, 1, 2)
            x = x.to(torch.float32).to(device)
            y0, y1, y2 = (y[0].to(device), y[1].to(device), y[2].to(device))
            preds = model(x)
            loss_val = (loss_f(preds[0], y0, scaled_anchors[0]) + loss_f(preds[1], y1, scaled_anchors[1]) + loss_f(preds[2], y2, scaled_anchors[2]))
            class_acc, noobj_acc, obj_acc = class_accuracy(preds, y, arguments["conf_thresh"])
            
    return (float(loss_val), class_acc, noobj_acc, obj_acc)

def main(arguments):
    if arguments["image_size"] == 608:
       anchors = [
       [(0.23, 0.18), (0.32, 0.40), (0.75, 0.66)],
       [(0.06, 0.12), (0.12, 0.09), (0.12, 0.24)],
       [(0.02, 0.03), (0.03, 0.06), (0.07, 0.05)], 
       ]
       S = [19, 38, 76]

    if arguments["image_size"] == 416:
       anchors = [
       [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
       [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
       [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
       ]
       S = [13, 26, 52]

    last_run = 0
    last_epoch = 0
    train_loss_lst = []
    test_loss_lst = []
    train_top1acc_lst = []
    test_top1acc_lst = []
    train_top5acc_lst = []
    test_top5acc_lst = []
    train_map_lst = []
    test_map_lst = []
    test_yolo_noobj_acc_lst = []
    test_yolo_obj_acc_lst = []
    test_yolo_class_acc_lst = []
    train_yolo_noobj_acc_lst = []
