import torch
import torchvision
import torchvision.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from models.rnn import SimpleRNN, GruRNN, LstmRNN, UrLstmRNN, HippoRNN
from models.s4.model import S4Model
from models.s4.s4_model import S4Model as S4
from models.transformer import Transformer
from models.hippo import Hippo
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import write_results
import numpy as np
from train.rnn_train_fn import train, evaluate
from train.hippo_train_fn import trainhippo, evaluatehippo
from torch.cuda.amp import GradScaler

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device Name: {torch.cuda.get_device_name(device)}" if device.type == 'cuda' else "CPU")
#torch.autograd.set_detect_anomaly(True)

data_dir =  'data/'

model_names = ['rnn', 
                'gru',
                'lstm',
                'urlstm',
                'hippo',
                'gatedhippo',
                'transformer',
                's4']

trains_normal = ['Simple Rnn', 
                'Gru Rnn',
                'Lstm Rnn',
                'Ur Lstm Rnn', 
                'Hippo Rnn', 
                'Transformer',
                'S4']

trains_hippo = ['Simple Hippo']


dataset_names = ['mnist']

floatTypes = [
  "weight_decay", 
  "lr",
  "momentum",
  "min_lr"
]

intTypes = [
  "sequence_length", 
  "input_size",
  "hidden_size",
  "output_size",
  "nclasses",
  "batch_size",
  "nepochs",
  "nruns",
  "nworkers",
  "N",
]

boolTypes = [
  "save_model",
  "cosine_anneal",
  "exponential_decay",
  "one_cycle",
]

def initialize_with_args(_arguments):
  arguments = {
      "model_name": "rnn",
      "dataset_name": "mnist",
      "save_model": False,
      "weight_decay": 0.0005,
      "sequence_length": 784,
      "input_size": 1,
      "hidden_size": 512,
      "output_size": 10,
      "nclasses": 10,
      "batch_size": 64,
      "nepochs": 50,
      "nruns": 1,
      "nworkers": 2,
      "N": 512,
      "init_rnn": False,
      "init_grurnn": False,
      "init_lstmrnn": False,
      "init_urlstmrnn": False,
      "init_hippo": False,
      "init_hippornn": False,
      "init_transformer": False,
      "init_s4": False,
      "mnist": False,
      "current_model": "",
      "lr": 1e-4, # inital learning rate
      "momentum": 0.9, # default val in adam pytorch
      "min_lr": 1e-6,  # for cosine anneal
      "cosine_anneal": False,
      "exponential_decay": False,
      "one_cycle": False,
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
    print(f"model name {arguments['model_name']} was not found, use simple, gru, lstm, hippo or gatedhippo.")
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
  elif arguments["model_name"] == "transformer":
    arguments["init_transformer"] = True
    arguments["current_model"] = model_names[6]
    arguments["path_cpt_file"] = 'cpts/{}_rnn_mnist.cpt'.format(arguments["current_model"])
    arguments["model_name"] = 'Transformer'
  elif arguments["model_name"] == "s4":
    arguments["init_s4"] = True
    arguments["current_model"] = model_names[7]
    arguments["path_cpt_file"] = 'cpts/{}_rnn_mnist.cpt'.format(arguments["current_model"])
    arguments["model_name"] = 'S4'

  if arguments["dataset_name"] == "mnist":
      arguments["mnist"] = True

  main(arguments)

def main(arguments):

    last_run = 0
    start_epoch = 0
    train_loss_lst = []
    test_loss_lst = []
    train_top1acc_lst = []
    test_top1acc_lst = []
    train_top5acc_lst = []
    test_top5acc_lst = []

    for run in range(last_run, arguments["nruns"]):
            
        if arguments["init_rnn"] == True: 
            model = SimpleRNN(input_size = arguments['input_size'], hidden_size = arguments["hidden_size"], output_size = arguments["output_size"], activation = 'relu').to(device)

        elif arguments["init_grurnn"] == True:
            model = GruRNN(input_size = arguments['input_size'], hidden_size = arguments["hidden_size"], output_size = arguments["output_size"]).to(device)

        elif arguments["init_lstmrnn"] == True:
            model = LstmRNN(input_size = arguments['input_size'], hidden_size = arguments["hidden_size"], output_size = arguments["output_size"]).to(device)

        elif arguments["init_urlstmrnn"] == True:
            model = UrLstmRNN(input_size = arguments['input_size'], hidden_size = arguments["hidden_size"], output_size = arguments["output_size"]).to(device)

        elif arguments["init_hippo"] == True:
            model = Hippo(N = arguments['N'], maxlength = arguments['sequence_length'], output_size = arguments["output_size"]).to(device)

        elif arguments["init_hippornn"] == True:
            model = HippoRNN(input_size = arguments['input_size'], hidden_size = arguments['hidden_size'], output_size = arguments["output_size"], maxlength=(28*28) // arguments["input_size"]).to(device)

        elif arguments["init_transformer"] == True:
            model = Transformer(input_size = arguments['input_size'], model_dim = 128, num_heads = 4, num_layers = 2, dim_feedforward= 512, output_size = arguments["output_size"], seq_len = 28*28).to(device)

        elif arguments["init_s4"] == True:
            # model = S4Model(d_input=arguments['input_size'], d_model=128*2, d_output=10, n_blocks=4, n=64, l_max=784, p_dropout=0.1, collapse=True, norm_type="layer", wavelet_tform=True).to(device)
            model = S4(d_input = arguments['input_size'], d_output=arguments['output_size'], d_model=128, n_layers=2, dropout=0.2,).to(device)

        if arguments['model_name'] in trains_normal:
            print(model)
            print(f'Run {run + 1}/{arguments["nruns"]}: {arguments["model_name"]} initalised with {arguments["hidden_size"]} number of hidden neurons.')
            print(f'Learning rate scheduler: {"Cosine Annealing" if arguments["cosine_anneal"] else "Exponential Decay" if arguments["exponential_decay"] else "Constant"}')
            print(f'Training parameters: \n {arguments["nepochs"]} : epochs. \n {arguments["input_size"]} : input size. \n {arguments["hidden_size"]} : hidden size. \n {arguments["output_size"]} : output size. \n {arguments["lr"]} : learning rate. \n {arguments["weight_decay"]} : weight decay. \n {arguments["momentum"]} : momentum.')
        
        if arguments['model_name'] in trains_hippo:
            print(model)
            print(f'Run {run + 1}/{arguments["nruns"]}: {arguments["model_name"]} initalised with {arguments["N"]} number of coefficents.' )
            print(f'Learning rate scheduler: {"Cosine Annealing" if arguments["cosine_anneal"] else "Exponential Decay" if arguments["exponential_decay"] else "Constant"}')
            print(f'Training parameters: \n {arguments["nepochs"]} : epochs. \n {arguments["input_size"]} : input size. \n {arguments["N"]} : coefficents. \n {arguments["output_size"]} : output size. \n {arguments["lr"]} : learning rate. \n {arguments["weight_decay"]} : weight decay. \n {arguments["momentum"]} : momentum.')
        
        optimizer = optim.Adam(model.parameters(), lr = arguments["lr"], weight_decay = arguments["weight_decay"], betas = (arguments["momentum"], 0.999))

        # ensure that lists are empty when a new model is initalised so
        # that lists from a previous run do not interfere with storage
        train_loss_lst = []
        test_loss_lst = []
        train_top1acc_lst = []
        test_top1acc_lst = []
        train_top5acc_lst = []
        test_top5acc_lst = []
        
        if arguments["mnist"] == True:
            train_dataset = torchvision.datasets.MNIST(root = data_dir,
                                                train = True, 
                                                transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))]),
                                                download = True)

            test_dataset = torchvision.datasets.MNIST(root =  data_dir,
                                                train = False, 
                                                transform = T.Compose([T.ToTensor(), T.Normalize((0.5,), (0.5,))]))
            loss_f = nn.CrossEntropyLoss()

        # we drop the last batch to ensure each batch has the same size
        train_loader = DataLoader(dataset = train_dataset, num_workers = arguments["nworkers"],
                                            batch_size = arguments["batch_size"],
                                            shuffle = True, drop_last = False)
        
        test_loader = DataLoader(dataset = test_dataset, num_workers = arguments["nworkers"],
                                            batch_size = arguments["batch_size"],
                                            shuffle = False, drop_last = False)

        if arguments["cosine_anneal"] == True:
           scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = arguments["nepochs"], eta_min = arguments["min_lr"])

        if arguments["exponential_decay"] == True:
           scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        
        if arguments["one_cycle"] == True:
           scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr = arguments["lr"], epochs = arguments["nepochs"], steps_per_epoch = len(train_loader), anneal_strategy = "cos")
        
        for epoch in range(start_epoch, arguments["nepochs"]):
            # we dont compute train map to save compute but compute test_map every 10 Epochs
            # None is changed to the computed map and if not we print an empty string
            train_map_val = None
            train_recall_val = None
            train_precision_val = None
            test_map_val = None
            test_recall_val = None
            test_precision_val = None
            # calling scheduler step before optimizer step will trigger a warning
            # however since we adjust learning rate based on epoch and then want to train it can be ignored.
            if (arguments["cosine_anneal"] == True or arguments["exponential_decay"] == True):
                scheduler.step()
                        
            if arguments['model_name'] in trains_normal:
                # train
                train_loss_value, train_top1acc_value, train_top5acc_value = train(arguments, train_loader, model, optimizer, loss_f)
                train_loss_lst.append(train_loss_value)
                train_top1acc_lst.append(train_top1acc_value)
                train_top5acc_lst.append(train_top5acc_value)

                # test 
                test_loss_value, test_top1acc_value, test_top5acc_value = evaluate(arguments, test_loader, model, loss_f)
                test_loss_lst.append(test_loss_value)
                test_top1acc_lst.append(test_top1acc_value)
                test_top5acc_lst.append(test_top5acc_value)
            

            if arguments['model_name'] in trains_hippo:
                # train
                train_loss_value, train_top1acc_value, train_top5acc_value = trainhippo(arguments, train_loader, model, optimizer, loss_f)
                train_loss_lst.append(train_loss_value)
                train_top1acc_lst.append(train_top1acc_value)
                train_top5acc_lst.append(train_top5acc_value)
              
                # test 
                test_loss_value, test_top1acc_value, test_top5acc_value = evaluatehippo(arguments, test_loader, model, loss_f)
                test_loss_lst.append(test_loss_value)
                test_top1acc_lst.append(test_top1acc_value)
                test_top5acc_lst.append(test_top5acc_value)

            print(f"Epoch:{epoch + 1}   Train[Loss:{train_loss_value} Top1 Acc:{train_top1acc_value}  Top5 Acc:{train_top5acc_value}]")
            print(f"Epoch:{epoch + 1}   Test[Loss:{test_loss_value}   Top1 Acc:{test_top1acc_value}   Top5 Acc:{test_top5acc_value}]")

            if arguments["cosine_anneal"] == True or arguments["exponential_decay"] == True:
                scheduler.step()
            
            if arguments["save_model"] == True and ((epoch + 1) % 10) == 0 or epoch + 1 == arguments["nepochs"]:
                torch.save({
                    'epoch': epoch,
                    'run': run,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, arguments["path_cpt_file"])
                print(f"Checkpoint and evaluation at epoch {epoch + 1} stored")
                

                if arguments['model_name'] in trains_normal:
                    file_prefix = arguments["current_model"]
                    rnn_metrics = ['train_loss', 'train_top1acc', 'train_top5acc', 'test_loss', 'test_top1acc', 'test_top5acc']
                    write_results(file_prefix, rnn_metrics, [train_loss_lst, train_top1acc_lst, train_top5acc_lst,
                                                            test_loss_lst, test_top1acc_lst, test_top5acc_lst], run)

                if arguments['model_name'] in trains_hippo:
                    file_prefix = arguments["current_model"]
                    hippo_metrics = ['train_loss', 'train_top1acc', 'train_top5acc', 'test_loss', 'test_top1acc', 'test_top5acc']
                    write_results(file_prefix, hippo_metrics, [train_loss_lst, train_top1acc_lst, train_top5acc_lst,
                                                            test_loss_lst, test_top1acc_lst, test_top5acc_lst], run)
                
            # if epoch has reached last epoch reset start_epoch variable to zero
            # to ensure that once we start another run we start at the first epoch
            if epoch == arguments["nepochs"] - 1:
                start_epoch = 0

# if __name__ == "__main__":
#     main()
