import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from scipy import stats
from utils import strip_square_brackets

epochs = 50

# train loss
strip_square_brackets("results/rnn_train_loss_run1.txt")
strip_square_brackets("results/rnn_train_loss_run2.txt")
strip_square_brackets("results/rnn_train_loss_run3.txt")
strip_square_brackets("results/rnn_train_loss_run4.txt")
strip_square_brackets("results/rnn_train_loss_run5.txt")

strip_square_brackets("results/grurnn_train_loss_run1.txt")
strip_square_brackets("results/grurnn_train_loss_run2.txt")
strip_square_brackets("results/grurnn_train_loss_run3.txt")
strip_square_brackets("results/grurnn_train_loss_run4.txt")
strip_square_brackets("results/grurnn_train_loss_run5.txt")

strip_square_brackets("results/lstmrnn_train_loss_run1.txt")
strip_square_brackets("results/lstmrnn_train_loss_run2.txt")
strip_square_brackets("results/lstmrnn_train_loss_run3.txt")
strip_square_brackets("results/lstmrnn_train_loss_run4.txt")
strip_square_brackets("results/lstmrnn_train_loss_run5.txt")

strip_square_brackets("results/urlstmrnn_train_loss_run1.txt")
strip_square_brackets("results/urlstmrnn_train_loss_run2.txt")
strip_square_brackets("results/urlstmrnn_train_loss_run3.txt")
strip_square_brackets("results/urlstmrnn_train_loss_run4.txt")
strip_square_brackets("results/urlstmrnn_train_loss_run5.txt")

strip_square_brackets("results/urlstmrnn_train_loss_run1.txt")
strip_square_brackets("results/urlstmrnn_train_loss_run2.txt")
strip_square_brackets("results/urlstmrnn_train_loss_run3.txt")
strip_square_brackets("results/urlstmrnn_train_loss_run4.txt")
strip_square_brackets("results/urlstmrnn_train_loss_run5.txt")

strip_square_brackets("results/hippo_train_loss_run1.txt")
strip_square_brackets("results/hippo_train_loss_run2.txt")
strip_square_brackets("results/hippo_train_loss_run3.txt")
strip_square_brackets("results/hippo_train_loss_run4.txt")
strip_square_brackets("results/hippo_train_loss_run5.txt")

strip_square_brackets("results/gatedhippornn_train_loss_run1.txt")
strip_square_brackets("results/gatedhippornn_train_loss_run2.txt")
strip_square_brackets("results/gatedhippornn_train_loss_run3.txt")
strip_square_brackets("results/gatedhippornn_train_loss_run4.txt")
strip_square_brackets("results/gatedhippornn_train_loss_run5.txt")

# test loss
strip_square_brackets("results/rnn_test_loss_run1.txt")
strip_square_brackets("results/rnn_test_loss_run2.txt")
strip_square_brackets("results/rnn_test_loss_run3.txt")
strip_square_brackets("results/rnn_test_loss_run4.txt")
strip_square_brackets("results/rnn_test_loss_run5.txt")

strip_square_brackets("results/grurnn_test_loss_run1.txt")
strip_square_brackets("results/grurnn_test_loss_run2.txt")
strip_square_brackets("results/grurnn_test_loss_run3.txt")
strip_square_brackets("results/grurnn_test_loss_run4.txt")
strip_square_brackets("results/grurnn_test_loss_run5.txt")

strip_square_brackets("results/lstmrnn_test_loss_run1.txt")
strip_square_brackets("results/lstmrnn_test_loss_run2.txt")
strip_square_brackets("results/lstmrnn_test_loss_run3.txt")
strip_square_brackets("results/lstmrnn_test_loss_run4.txt")
strip_square_brackets("results/lstmrnn_test_loss_run5.txt")

strip_square_brackets("results/urlstmrnn_test_loss_run1.txt")
strip_square_brackets("results/urlstmrnn_test_loss_run2.txt")
strip_square_brackets("results/urlstmrnn_test_loss_run3.txt")
strip_square_brackets("results/urlstmrnn_test_loss_run4.txt")
strip_square_brackets("results/urlstmrnn_test_loss_run5.txt")

strip_square_brackets("results/hippo_test_loss_run1.txt")
strip_square_brackets("results/hippo_test_loss_run2.txt")
strip_square_brackets("results/hippo_test_loss_run3.txt")
strip_square_brackets("results/hippo_test_loss_run4.txt")
strip_square_brackets("results/hippo_test_loss_run5.txt")

strip_square_brackets("results/gatedhippornn_test_loss_run1.txt")
strip_square_brackets("results/gatedhippornn_test_loss_run2.txt")
strip_square_brackets("results/gatedhippornn_test_loss_run3.txt")
strip_square_brackets("results/gatedhippornn_test_loss_run4.txt")
strip_square_brackets("results/gatedhippornn_test_loss_run5.txt")

# train top 1 accuracy
strip_square_brackets("results/rnn_train_top1acc_run1.txt")
strip_square_brackets("results/rnn_train_top1acc_run2.txt")
strip_square_brackets("results/rnn_train_top1acc_run3.txt")
strip_square_brackets("results/rnn_train_top1acc_run4.txt")
strip_square_brackets("results/rnn_train_top1acc_run5.txt")

strip_square_brackets("results/grurnn_train_top1acc_run1.txt")
strip_square_brackets("results/grurnn_train_top1acc_run2.txt")
strip_square_brackets("results/grurnn_train_top1acc_run3.txt")
strip_square_brackets("results/grurnn_train_top1acc_run4.txt")
strip_square_brackets("results/grurnn_train_top1acc_run5.txt")

strip_square_brackets("results/lstmrnn_train_top1acc_run1.txt")
strip_square_brackets("results/lstmrnn_train_top1acc_run2.txt")
strip_square_brackets("results/lstmrnn_train_top1acc_run3.txt")
strip_square_brackets("results/lstmrnn_train_top1acc_run4.txt")
strip_square_brackets("results/lstmrnn_train_top1acc_run5.txt")

strip_square_brackets("results/urlstmrnn_train_top1acc_run1.txt")
strip_square_brackets("results/urlstmrnn_train_top1acc_run2.txt")
strip_square_brackets("results/urlstmrnn_train_top1acc_run3.txt")
strip_square_brackets("results/urlstmrnn_train_top1acc_run4.txt")
strip_square_brackets("results/urlstmrnn_train_top1acc_run5.txt")

strip_square_brackets("results/hippo_train_top1acc_run1.txt")
strip_square_brackets("results/hippo_train_top1acc_run2.txt")
strip_square_brackets("results/hippo_train_top1acc_run3.txt")
strip_square_brackets("results/hippo_train_top1acc_run4.txt")
strip_square_brackets("results/hippo_train_top1acc_run5.txt")

strip_square_brackets("results/gatedhippornn_train_top1acc_run1.txt")
strip_square_brackets("results/gatedhippornn_train_top1acc_run2.txt")
strip_square_brackets("results/gatedhippornn_train_top1acc_run3.txt")
strip_square_brackets("results/gatedhippornn_train_top1acc_run4.txt")
strip_square_brackets("results/gatedhippornn_train_top1acc_run5.txt")

# test top 1 accuracy
strip_square_brackets("results/rnn_test_top1acc_run1.txt")
strip_square_brackets("results/rnn_test_top1acc_run2.txt")
strip_square_brackets("results/rnn_test_top1acc_run3.txt")
strip_square_brackets("results/rnn_test_top1acc_run4.txt")
strip_square_brackets("results/rnn_test_top1acc_run5.txt")

strip_square_brackets("results/grurnn_test_top1acc_run1.txt")
strip_square_brackets("results/grurnn_test_top1acc_run2.txt")
strip_square_brackets("results/grurnn_test_top1acc_run3.txt")
strip_square_brackets("results/grurnn_test_top1acc_run4.txt")
strip_square_brackets("results/grurnn_test_top1acc_run5.txt")

strip_square_brackets("results/lstmrnn_test_top1acc_run1.txt")
strip_square_brackets("results/lstmrnn_test_top1acc_run2.txt")
strip_square_brackets("results/lstmrnn_test_top1acc_run3.txt")
strip_square_brackets("results/lstmrnn_test_top1acc_run4.txt")
strip_square_brackets("results/lstmrnn_test_top1acc_run5.txt")

strip_square_brackets("results/urlstmrnn_test_top1acc_run1.txt")
strip_square_brackets("results/urlstmrnn_test_top1acc_run2.txt")
strip_square_brackets("results/urlstmrnn_test_top1acc_run3.txt")
strip_square_brackets("results/urlstmrnn_test_top1acc_run4.txt")
strip_square_brackets("results/urlstmrnn_test_top1acc_run5.txt")

strip_square_brackets("results/hippo_test_top1acc_run1.txt")
strip_square_brackets("results/hippo_test_top1acc_run2.txt")
strip_square_brackets("results/hippo_test_top1acc_run3.txt")
strip_square_brackets("results/hippo_test_top1acc_run4.txt")
strip_square_brackets("results/hippo_test_top1acc_run5.txt")

strip_square_brackets("results/gatedhippornn_test_top1acc_run1.txt")
strip_square_brackets("results/gatedhippornn_test_top1acc_run2.txt")
strip_square_brackets("results/gatedhippornn_test_top1acc_run3.txt")
strip_square_brackets("results/gatedhippornn_test_top1acc_run4.txt")
strip_square_brackets("results/gatedhippornn_test_top1acc_run5.txt")

# train loss
train_rnn_loss_run1 = genfromtxt("results/rnn_train_loss_run1.txt", delimiter=',')[:50]
train_rnn_loss_run2 = genfromtxt("results/rnn_train_loss_run2.txt", delimiter=',')[:50]
train_rnn_loss_run3 = genfromtxt("results/rnn_train_loss_run3.txt", delimiter=',')[:50]
train_rnn_loss_run4 = genfromtxt("results/rnn_train_loss_run4.txt", delimiter=',')[:50]
train_rnn_loss_run5 = genfromtxt("results/rnn_train_loss_run5.txt", delimiter=',')[:50]
train_rnn_loss_runs = np.vstack((train_rnn_loss_run1, train_rnn_loss_run2, train_rnn_loss_run3,
                  train_rnn_loss_run4, train_rnn_loss_run5))

train_grurnn_loss_run1 = genfromtxt("results/grurnn_train_loss_run1.txt", delimiter=',')[:50]
train_grurnn_loss_run2 = genfromtxt("results/grurnn_train_loss_run2.txt", delimiter=',')[:50]
train_grurnn_loss_run3 = genfromtxt("results/grurnn_train_loss_run3.txt", delimiter=',')[:50]
train_grurnn_loss_run4 = genfromtxt("results/grurnn_train_loss_run4.txt", delimiter=',')[:50]
train_grurnn_loss_run5 = genfromtxt("results/grurnn_train_loss_run5.txt", delimiter=',')[:50]
train_grurnn_loss_runs = np.vstack((train_grurnn_loss_run1, train_grurnn_loss_run2, train_grurnn_loss_run3,
                  train_grurnn_loss_run4, train_grurnn_loss_run5))


train_lstmrnn_loss_run1 = genfromtxt("results/lstmrnn_train_loss_run1.txt", delimiter=',')[:50]
train_lstmrnn_loss_run2 = genfromtxt("results/lstmrnn_train_loss_run2.txt", delimiter=',')[:50]
train_lstmrnn_loss_run3 = genfromtxt("results/lstmrnn_train_loss_run3.txt", delimiter=',')[:50]
train_lstmrnn_loss_run4 = genfromtxt("results/lstmrnn_train_loss_run4.txt", delimiter=',')[:50]
train_lstmrnn_loss_run5 = genfromtxt("results/lstmrnn_train_loss_run5.txt", delimiter=',')[:50]
train_lstmrnn_loss_runs = np.vstack((train_lstmrnn_loss_run1, train_lstmrnn_loss_run2, train_lstmrnn_loss_run3,
                  train_lstmrnn_loss_run4, train_lstmrnn_loss_run5))

train_urlstmrnn_loss_run1 = genfromtxt("results/urlstmrnn_train_loss_run1.txt", delimiter=',')[:50]
train_urlstmrnn_loss_run2 = genfromtxt("results/urlstmrnn_train_loss_run2.txt", delimiter=',')[:50]
train_urlstmrnn_loss_run3 = genfromtxt("results/urlstmrnn_train_loss_run3.txt", delimiter=',')[:50]
train_urlstmrnn_loss_run4 = genfromtxt("results/urlstmrnn_train_loss_run4.txt", delimiter=',')[:50]
train_urlstmrnn_loss_run5 = genfromtxt("results/urlstmrnn_train_loss_run5.txt", delimiter=',')[:50]
train_urlstmrnn_loss_runs = np.vstack((train_urlstmrnn_loss_run1, train_urlstmrnn_loss_run2, train_urlstmrnn_loss_run3,
                 train_urlstmrnn_loss_run4, train_urlstmrnn_loss_run5))

train_hippo_loss_run1 = genfromtxt("results/hippo_train_loss_run1.txt", delimiter=',')[:50]
train_hippo_loss_run2 = genfromtxt("results/hippo_train_loss_run2.txt", delimiter=',')[:50]
train_hippo_loss_run3 = genfromtxt("results/hippo_train_loss_run3.txt", delimiter=',')[:50]
train_hippo_loss_run4 = genfromtxt("results/hippo_train_loss_run4.txt", delimiter=',')[:50]
train_hippo_loss_run5 = genfromtxt("results/hippo_train_loss_run5.txt", delimiter=',')[:50]
train_hippo_loss_runs = np.vstack((train_hippo_loss_run1, train_hippo_loss_run2, train_hippo_loss_run3,
                 train_hippo_loss_run4, train_hippo_loss_run5))

train_gatedhippornn_loss_run1 = genfromtxt("results/gatedhippornn_train_loss_run1.txt", delimiter=',')[:50]
train_gatedhippornn_loss_run2 = genfromtxt("results/gatedhippornn_train_loss_run2.txt", delimiter=',')[:50]
train_gatedhippornn_loss_run3 = genfromtxt("results/gatedhippornn_train_loss_run3.txt", delimiter=',')[:50]
train_gatedhippornn_loss_run4 = genfromtxt("results/gatedhippornn_train_loss_run4.txt", delimiter=',')[:50]
train_gatedhippornn_loss_run5 = genfromtxt("results/gatedhippornn_train_loss_run5.txt", delimiter=',')[:50]
train_gatedhippornn_loss_runs = np.vstack((train_gatedhippornn_loss_run1, train_gatedhippornn_loss_run2, train_gatedhippornn_loss_run3,
                 train_gatedhippornn_loss_run4, train_gatedhippornn_loss_run5))

# test loss
test_rnn_loss_run1 = genfromtxt("results/rnn_test_loss_run1.txt", delimiter=',')[:50]
test_rnn_loss_run2 = genfromtxt("results/rnn_test_loss_run2.txt", delimiter=',')[:50]
test_rnn_loss_run3 = genfromtxt("results/rnn_test_loss_run3.txt", delimiter=',')[:50]
test_rnn_loss_run4 = genfromtxt("results/rnn_test_loss_run4.txt", delimiter=',')[:50]
test_rnn_loss_run5 = genfromtxt("results/rnn_test_loss_run5.txt", delimiter=',')[:50]
test_rnn_loss_runs = np.vstack((test_rnn_loss_run1, test_rnn_loss_run2, test_rnn_loss_run3,
                 test_rnn_loss_run4, test_rnn_loss_run5))

test_grurnn_loss_run1 = genfromtxt("results/grurnn_test_loss_run1.txt", delimiter=',')[:50]
test_grurnn_loss_run2 = genfromtxt("results/grurnn_test_loss_run2.txt", delimiter=',')[:50]
test_grurnn_loss_run3 = genfromtxt("results/grurnn_test_loss_run3.txt", delimiter=',')[:50]
test_grurnn_loss_run4 = genfromtxt("results/grurnn_test_loss_run4.txt", delimiter=',')[:50]
test_grurnn_loss_run5 = genfromtxt("results/grurnn_test_loss_run5.txt", delimiter=',')[:50]
test_grurnn_loss_runs = np.vstack((test_grurnn_loss_run1, test_grurnn_loss_run2, test_grurnn_loss_run3,
                 test_grurnn_loss_run4, test_grurnn_loss_run5))

test_lstmrnn_loss_run1 = genfromtxt("results/lstmrnn_test_loss_run1.txt", delimiter=',')[:50]
test_lstmrnn_loss_run2 = genfromtxt("results/lstmrnn_test_loss_run2.txt", delimiter=',')[:50]
test_lstmrnn_loss_run3 = genfromtxt("results/lstmrnn_test_loss_run3.txt", delimiter=',')[:50]
test_lstmrnn_loss_run4 = genfromtxt("results/lstmrnn_test_loss_run4.txt", delimiter=',')[:50]
test_lstmrnn_loss_run5 = genfromtxt("results/lstmrnn_test_loss_run5.txt", delimiter=',')[:50]
test_lstmrnn_loss_runs = np.vstack((test_lstmrnn_loss_run1, test_lstmrnn_loss_run2, test_lstmrnn_loss_run3,
                 test_lstmrnn_loss_run4, test_lstmrnn_loss_run5))

test_urlstmrnn_loss_run1 = genfromtxt("results/urlstmrnn_test_loss_run1.txt", delimiter=',')[:50]
test_urlstmrnn_loss_run2 = genfromtxt("results/urlstmrnn_test_loss_run2.txt", delimiter=',')[:50]
test_urlstmrnn_loss_run3 = genfromtxt("results/urlstmrnn_test_loss_run3.txt", delimiter=',')[:50]
test_urlstmrnn_loss_run4 = genfromtxt("results/urlstmrnn_test_loss_run4.txt", delimiter=',')[:50]
test_urlstmrnn_loss_run5 = genfromtxt("results/urlstmrnn_test_loss_run5.txt", delimiter=',')[:50]
test_urlstmrnn_loss_runs = np.vstack((test_urlstmrnn_loss_run1, test_urlstmrnn_loss_run2, test_urlstmrnn_loss_run3,
                 test_urlstmrnn_loss_run4, test_urlstmrnn_loss_run5))

test_hippo_loss_run1 = genfromtxt("results/hippo_test_loss_run1.txt", delimiter=',')[:50]
test_hippo_loss_run2 = genfromtxt("results/hippo_test_loss_run2.txt", delimiter=',')[:50]
test_hippo_loss_run3 = genfromtxt("results/hippo_test_loss_run3.txt", delimiter=',')[:50]
test_hippo_loss_run4 = genfromtxt("results/hippo_test_loss_run4.txt", delimiter=',')[:50]
test_hippo_loss_run5 = genfromtxt("results/hippo_test_loss_run5.txt", delimiter=',')[:50]
test_hippo_loss_runs = np.vstack((test_hippo_loss_run1, test_hippo_loss_run2, test_hippo_loss_run3,
                 test_hippo_loss_run4, test_hippo_loss_run5))

test_gatedhippornn_loss_run1 = genfromtxt("results/gatedhippornn_test_loss_run1.txt", delimiter=',')[:50]
test_gatedhippornn_loss_run2 = genfromtxt("results/gatedhippornn_test_loss_run2.txt", delimiter=',')[:50]
test_gatedhippornn_loss_run3 = genfromtxt("results/gatedhippornn_test_loss_run3.txt", delimiter=',')[:50]
test_gatedhippornn_loss_run4 = genfromtxt("results/gatedhippornn_test_loss_run4.txt", delimiter=',')[:50]
test_gatedhippornn_loss_run5 = genfromtxt("results/gatedhippornn_test_loss_run5.txt", delimiter=',')[:50]
test_gatedhippornn_loss_runs = np.vstack((test_gatedhippornn_loss_run1, test_gatedhippornn_loss_run2, test_gatedhippornn_loss_run3,
                 test_gatedhippornn_loss_run4, test_gatedhippornn_loss_run5))

# train top 1 accuracy
train_rnn_top1acc_run1 = genfromtxt("results/rnn_train_top1acc_run1.txt", delimiter=',')[:50]
train_rnn_top1acc_run2 = genfromtxt("results/rnn_train_top1acc_run2.txt", delimiter=',')[:50]
train_rnn_top1acc_run3 = genfromtxt("results/rnn_train_top1acc_run3.txt", delimiter=',')[:50]
train_rnn_top1acc_run4 = genfromtxt("results/rnn_train_top1acc_run4.txt", delimiter=',')[:50]
train_rnn_top1acc_run5 = genfromtxt("results/rnn_train_top1acc_run5.txt", delimiter=',')[:50]
train_rnn_top1acc_runs = np.vstack((train_rnn_top1acc_run1, train_rnn_top1acc_run2, train_rnn_top1acc_run3,
                  train_rnn_top1acc_run4, train_rnn_top1acc_run5))

train_grurnn_top1acc_run1 = genfromtxt("results/grurnn_train_top1acc_run1.txt", delimiter=',')[:50]
train_grurnn_top1acc_run2 = genfromtxt("results/grurnn_train_top1acc_run2.txt", delimiter=',')[:50]
train_grurnn_top1acc_run3 = genfromtxt("results/grurnn_train_top1acc_run3.txt", delimiter=',')[:50]
train_grurnn_top1acc_run4 = genfromtxt("results/grurnn_train_top1acc_run4.txt", delimiter=',')[:50]
train_grurnn_top1acc_run5 = genfromtxt("results/grurnn_train_top1acc_run5.txt", delimiter=',')[:50]
train_grurnn_top1acc_runs = np.vstack((train_grurnn_top1acc_run1, train_grurnn_top1acc_run2, train_grurnn_top1acc_run3,
                  train_grurnn_top1acc_run4, train_grurnn_top1acc_run5))

train_lstmrnn_top1acc_run1 = genfromtxt("results/lstmrnn_train_top1acc_run1.txt", delimiter=',')[:50]
train_lstmrnn_top1acc_run2 = genfromtxt("results/lstmrnn_train_top1acc_run2.txt", delimiter=',')[:50]
train_lstmrnn_top1acc_run3 = genfromtxt("results/lstmrnn_train_top1acc_run3.txt", delimiter=',')[:50]
train_lstmrnn_top1acc_run4 = genfromtxt("results/lstmrnn_train_top1acc_run4.txt", delimiter=',')[:50]
train_lstmrnn_top1acc_run5 = genfromtxt("results/lstmrnn_train_top1acc_run5.txt", delimiter=',')[:50]
train_lstmrnn_top1acc_runs = np.vstack((train_lstmrnn_top1acc_run1, train_lstmrnn_top1acc_run2, train_lstmrnn_top1acc_run3,
                  train_lstmrnn_top1acc_run4, train_lstmrnn_top1acc_run5))


train_urlstmrnn_top1acc_run1 = genfromtxt("results/urlstmrnn_train_top1acc_run1.txt", delimiter=',')[:50]
train_urlstmrnn_top1acc_run2 = genfromtxt("results/urlstmrnn_train_top1acc_run2.txt", delimiter=',')[:50]
train_urlstmrnn_top1acc_run3 = genfromtxt("results/urlstmrnn_train_top1acc_run3.txt", delimiter=',')[:50]
train_urlstmrnn_top1acc_run4 = genfromtxt("results/urlstmrnn_train_top1acc_run4.txt", delimiter=',')[:50]
train_urlstmrnn_top1acc_run5 = genfromtxt("results/urlstmrnn_train_top1acc_run5.txt", delimiter=',')[:50]
train_urlstmrnn_top1acc_runs = np.vstack((train_urlstmrnn_top1acc_run1, train_urlstmrnn_top1acc_run2, train_urlstmrnn_top1acc_run3,
                  train_urlstmrnn_top1acc_run4, train_urlstmrnn_top1acc_run5))

train_hippo_top1acc_run1 = genfromtxt("results/hippo_train_top1acc_run1.txt", delimiter=',')[:50]
train_hippo_top1acc_run2 = genfromtxt("results/hippo_train_top1acc_run2.txt", delimiter=',')[:50]
train_hippo_top1acc_run3 = genfromtxt("results/hippo_train_top1acc_run3.txt", delimiter=',')[:50]
train_hippo_top1acc_run4 = genfromtxt("results/hippo_train_top1acc_run4.txt", delimiter=',')[:50]
train_hippo_top1acc_run5 = genfromtxt("results/hippo_train_top1acc_run5.txt", delimiter=',')[:50]
train_hippo_top1acc_runs = np.vstack((train_hippo_top1acc_run1, train_hippo_top1acc_run2, train_hippo_top1acc_run3,
                  train_hippo_top1acc_run4, train_hippo_top1acc_run5))

train_gatedhippornn_top1acc_run1 = genfromtxt("results/gatedhippornn_train_top1acc_run1.txt", delimiter=',')[:50]
train_gatedhippornn_top1acc_run2 = genfromtxt("results/gatedhippornn_train_top1acc_run2.txt", delimiter=',')[:50]
train_gatedhippornn_top1acc_run3 = genfromtxt("results/gatedhippornn_train_top1acc_run3.txt", delimiter=',')[:50]
train_gatedhippornn_top1acc_run4 = genfromtxt("results/gatedhippornn_train_top1acc_run4.txt", delimiter=',')[:50]
train_gatedhippornn_top1acc_run5 = genfromtxt("results/gatedhippornn_train_top1acc_run5.txt", delimiter=',')[:50]
train_gatedhippornn_top1acc_runs = np.vstack((train_gatedhippornn_top1acc_run1, train_gatedhippornn_top1acc_run2, train_gatedhippornn_top1acc_run3,
                  train_gatedhippornn_top1acc_run4, train_gatedhippornn_top1acc_run5))

# test top 1 accuracy
test_rnn_top1acc_run1 = genfromtxt("results/rnn_test_top1acc_run1.txt", delimiter=',')[:50]
test_rnn_top1acc_run2 = genfromtxt("results/rnn_test_top1acc_run2.txt", delimiter=',')[:50]
test_rnn_top1acc_run3 = genfromtxt("results/rnn_test_top1acc_run3.txt", delimiter=',')[:50]
test_rnn_top1acc_run4 = genfromtxt("results/rnn_test_top1acc_run4.txt", delimiter=',')[:50]
test_rnn_top1acc_run5 = genfromtxt("results/rnn_test_top1acc_run5.txt", delimiter=',')[:50]
test_rnn_top1acc_runs = np.vstack((test_rnn_top1acc_run1, test_rnn_top1acc_run2, test_rnn_top1acc_run3,
                  test_rnn_top1acc_run4, test_rnn_top1acc_run5))

test_grurnn_top1acc_run1 = genfromtxt("results/grurnn_test_top1acc_run1.txt", delimiter=',')[:50]
test_grurnn_top1acc_run2 = genfromtxt("results/grurnn_test_top1acc_run2.txt", delimiter=',')[:50]
test_grurnn_top1acc_run3 = genfromtxt("results/grurnn_test_top1acc_run3.txt", delimiter=',')[:50]
test_grurnn_top1acc_run4 = genfromtxt("results/grurnn_test_top1acc_run4.txt", delimiter=',')[:50]
test_grurnn_top1acc_run5 = genfromtxt("results/grurnn_test_top1acc_run5.txt", delimiter=',')[:50]
test_grurnn_top1acc_runs = np.vstack((test_grurnn_top1acc_run1, test_grurnn_top1acc_run2, test_grurnn_top1acc_run3,
                  test_grurnn_top1acc_run4, test_grurnn_top1acc_run5))

test_lstmrnn_top1acc_run1 = genfromtxt("results/lstmrnn_test_top1acc_run1.txt", delimiter=',')[:50]
test_lstmrnn_top1acc_run2 = genfromtxt("results/lstmrnn_test_top1acc_run2.txt", delimiter=',')[:50]
test_lstmrnn_top1acc_run3 = genfromtxt("results/lstmrnn_test_top1acc_run3.txt", delimiter=',')[:50]
test_lstmrnn_top1acc_run4 = genfromtxt("results/lstmrnn_test_top1acc_run4.txt", delimiter=',')[:50]
test_lstmrnn_top1acc_run5 = genfromtxt("results/lstmrnn_test_top1acc_run5.txt", delimiter=',')[:50]
test_lstmrnn_top1acc_runs = np.vstack((test_lstmrnn_top1acc_run1, test_lstmrnn_top1acc_run2, test_lstmrnn_top1acc_run3,
                  test_lstmrnn_top1acc_run4, test_lstmrnn_top1acc_run5))

test_urlstmrnn_top1acc_run1 = genfromtxt("results/urlstmrnn_test_top1acc_run1.txt", delimiter=',')[:50]
test_urlstmrnn_top1acc_run2 = genfromtxt("results/urlstmrnn_test_top1acc_run2.txt", delimiter=',')[:50]
test_urlstmrnn_top1acc_run3 = genfromtxt("results/urlstmrnn_test_top1acc_run3.txt", delimiter=',')[:50]
test_urlstmrnn_top1acc_run4 = genfromtxt("results/urlstmrnn_test_top1acc_run4.txt", delimiter=',')[:50]
test_urlstmrnn_top1acc_run5 = genfromtxt("results/urlstmrnn_test_top1acc_run5.txt", delimiter=',')[:50]
test_urlstmrnn_top1acc_runs = np.vstack((test_urlstmrnn_top1acc_run1, test_urlstmrnn_top1acc_run2, test_urlstmrnn_top1acc_run3,
                  test_urlstmrnn_top1acc_run4, test_urlstmrnn_top1acc_run5))

test_hippo_top1acc_run1 = genfromtxt("results/hippo_test_top1acc_run1.txt", delimiter=',')[:50]
test_hippo_top1acc_run2 = genfromtxt("results/hippo_test_top1acc_run2.txt", delimiter=',')[:50]
test_hippo_top1acc_run3 = genfromtxt("results/hippo_test_top1acc_run3.txt", delimiter=',')[:50]
test_hippo_top1acc_run4 = genfromtxt("results/hippo_test_top1acc_run4.txt", delimiter=',')[:50]
test_hippo_top1acc_run5 = genfromtxt("results/hippo_test_top1acc_run5.txt", delimiter=',')[:50]
test_hippo_top1acc_runs = np.vstack((test_hippo_top1acc_run1, test_hippo_top1acc_run2, test_hippo_top1acc_run3,
                  test_hippo_top1acc_run4, test_hippo_top1acc_run5))

test_gatedhippornn_top1acc_run1 = genfromtxt("results/gatedhippornn_test_top1acc_run1.txt", delimiter=',')[:50]
test_gatedhippornn_top1acc_run2 = genfromtxt("results/gatedhippornn_test_top1acc_run2.txt", delimiter=',')[:50]
test_gatedhippornn_top1acc_run3 = genfromtxt("results/gatedhippornn_test_top1acc_run3.txt", delimiter=',')[:50]
test_gatedhippornn_top1acc_run4 = genfromtxt("results/gatedhippornn_test_top1acc_run4.txt", delimiter=',')[:50]
test_gatedhippornn_top1acc_run5 = genfromtxt("results/gatedhippornn_test_top1acc_run5.txt", delimiter=',')[:50]
test_gatedhippornn_top1acc_runs = np.vstack((test_gatedhippornn_top1acc_run1, test_gatedhippornn_top1acc_run2, test_gatedhippornn_top1acc_run3,
                  test_gatedhippornn_top1acc_run4, test_gatedhippornn_top1acc_run5))

def compute_mean_std(data, std_multiplier=1):
    # Replace NaN values with 0
    data = np.nan_to_num(data, nan=0)

    # Compute the mean and standard deviation along the row axis
    mean_array = np.mean(data, axis=0)
    std_array = np.std(data, axis=0)

    # Calculate the lower and upper bounds as mean ± std_multiplier * standard deviation
    lower_bound = mean_array - std_multiplier * std_array
    upper_bound = mean_array + std_multiplier * std_array

    # Set values larger than 6 to 6
    mean_array = np.where(mean_array > 6, 6, mean_array)
    std_array = np.where(std_array > 6, 6, std_array)
    lower_bound = np.where(lower_bound > 6, 6, lower_bound)
    upper_bound = np.where(upper_bound > 6, 6, upper_bound)

    # Set values less than 0 to 0
    mean_array = np.where(mean_array < 0, 0, mean_array)
    std_array = np.where(std_array < 0, 0, std_array)
    lower_bound = np.where(lower_bound < 0, 0, lower_bound)
    upper_bound = np.where(upper_bound < 0, 0, upper_bound)

    return mean_array, std_array, lower_bound, upper_bound

# Combine the arrays into a 2D array

train_rnn_loss_mean, train_rnn_loss_std, train_rnn_loss_lower_bound, train_rnn_loss_upper_bound = compute_mean_std(train_rnn_loss_runs)
train_grurnn_loss_mean, train_grurnn_loss_std, train_grurnn_loss_lower_bound, train_grurnn_loss_upper_bound = compute_mean_std(train_grurnn_loss_runs)
train_lstmrnn_loss_mean, train_lstmrnn_loss_std, train_lstmrnn_loss_lower_bound, train_lstmrnn_loss_upper_bound = compute_mean_std(train_lstmrnn_loss_runs)
train_urlstmrnn_loss_mean, train_urlstmrnn_loss_std, train_urlstmrnn_loss_lower_bound, train_urlstmrnn_loss_upper_bound = compute_mean_std(train_urlstmrnn_loss_runs)
train_hippo_loss_mean, train_hippo_loss_std, train_hippo_loss_lower_bound, train_hippo_loss_upper_bound = compute_mean_std(train_hippo_loss_runs)
train_gatedhippornn_loss_mean, train_gatedhippornn_loss_std, train_gatedhippornn_loss_lower_bound, train_gatedhippornn_loss_upper_bound = compute_mean_std(train_gatedhippornn_loss_runs)

test_rnn_loss_mean, test_rnn_loss_std, test_rnn_loss_lower_bound, test_rnn_loss_upper_bound = compute_mean_std(test_rnn_loss_runs)
test_grurnn_loss_mean, test_grurnn_loss_std, test_grurnn_loss_lower_bound, test_grurnn_loss_upper_bound = compute_mean_std(test_grurnn_loss_runs)
test_lstmrnn_loss_mean, test_lstmrnn_loss_std, test_lstmrnn_loss_lower_bound, test_lstmrnn_loss_upper_bound = compute_mean_std(test_lstmrnn_loss_runs)
test_urlstmrnn_loss_mean, test_urlstmrnn_loss_std, test_urlstmrnn_loss_lower_bound, test_urlstmrnn_loss_upper_bound = compute_mean_std(test_urlstmrnn_loss_runs)
test_hippo_loss_mean, test_hippo_loss_std, test_hippo_loss_lower_bound, test_hippo_loss_upper_bound = compute_mean_std(test_hippo_loss_runs)
test_gatedhippornn_loss_mean, test_gatedhippornn_loss_std, test_gatedhippornn_loss_lower_bound, test_gatedhippornn_loss_upper_bound = compute_mean_std(test_gatedhippornn_loss_runs)


plt.figure(figsize=(13,5))
plt.subplot(1,2,1)
# Plot the mean line
plt.plot(train_rnn_loss_mean, label='Rnn')
# Plot the shaded area
plt.fill_between(np.arange(len(train_rnn_loss_mean)), train_rnn_loss_lower_bound, train_rnn_loss_upper_bound,
                 alpha=0.5,)

plt.plot(train_grurnn_loss_mean, label='Gru')
# Plot the shaded area
plt.fill_between(np.arange(len(train_grurnn_loss_mean)), train_grurnn_loss_lower_bound, train_grurnn_loss_upper_bound,
                 alpha=0.5,)

plt.plot(train_lstmrnn_loss_mean, label='Lstm')
# Plot the shaded area
plt.fill_between(np.arange(len(train_lstmrnn_loss_mean)), train_lstmrnn_loss_lower_bound, train_lstmrnn_loss_upper_bound,
                 alpha=0.5,)

plt.plot(train_urlstmrnn_loss_mean, label='UrLstm')
# Plot the shaded area
plt.fill_between(np.arange(len(train_urlstmrnn_loss_mean)), train_urlstmrnn_loss_lower_bound, train_urlstmrnn_loss_upper_bound,
                 alpha=0.5,)

plt.plot(train_hippo_loss_mean, label='Hippo')
# Plot the shaded area
plt.fill_between(np.arange(len(train_hippo_loss_mean)), train_hippo_loss_lower_bound, train_hippo_loss_upper_bound,
                 alpha=0.5,)

plt.plot(train_gatedhippornn_loss_mean, label='HippoRnn')
# Plot the shaded area
plt.fill_between(np.arange(len(train_gatedhippornn_loss_mean)), train_gatedhippornn_loss_lower_bound,train_gatedhippornn_loss_upper_bound,
                 alpha=0.5,)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training loss: Mean and standard deviation.')
plt.legend(frameon=False)

#plt.figure(figsize=(13,5))
plt.subplot(1,2,2)
# Plot the mean line
plt.plot(test_rnn_loss_mean, label='Rnn')
# Plot the shaded area
plt.fill_between(np.arange(len(test_rnn_loss_mean)), test_rnn_loss_lower_bound, test_rnn_loss_upper_bound,
                 alpha=0.5,)

plt.plot(test_grurnn_loss_mean, label='Gru')
# Plot the shaded area
plt.fill_between(np.arange(len(test_grurnn_loss_mean)), test_grurnn_loss_lower_bound, test_grurnn_loss_upper_bound,
                 alpha=0.5,)

plt.plot(test_lstmrnn_loss_mean, label='Lstm')
# Plot the shaded area
plt.fill_between(np.arange(len(test_lstmrnn_loss_mean)), test_lstmrnn_loss_lower_bound, test_lstmrnn_loss_upper_bound,
                 alpha=0.5,)

plt.plot(test_urlstmrnn_loss_mean, label='UrLstm')
# Plot the shaded area
plt.fill_between(np.arange(len(test_urlstmrnn_loss_mean)), test_urlstmrnn_loss_lower_bound, test_urlstmrnn_loss_upper_bound,
                 alpha=0.5,)

plt.plot(test_hippo_loss_mean, label='Hippo')
# Plot the shaded area
plt.fill_between(np.arange(len(test_hippo_loss_mean)), test_hippo_loss_lower_bound, test_hippo_loss_upper_bound,
                 alpha=0.5,)

plt.plot(test_gatedhippornn_loss_mean, label='HippoRnn')
# Plot the shaded area
plt.fill_between(np.arange(len(test_gatedhippornn_loss_mean)), test_gatedhippornn_loss_lower_bound, test_gatedhippornn_loss_upper_bound,
                 alpha=0.5,)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Test loss: Mean and standard deviation.')
plt.legend(frameon=False)
plt.savefig('figures/rnn_train_vs_test_loss.png')


def compute_mean_std(data, std_multiplier=1, upper_limit=1, lower_limit=0):
    # Replace NaN values with 0
    data = np.nan_to_num(data, nan=0)

    # Compute the mean and standard deviation along the row axis
    mean_array = np.mean(data, axis=0)
    std_array = np.std(data, axis=0)

    # Calculate the lower and upper bounds as mean ± std_multiplier * standard deviation
    lower_bound = mean_array - std_multiplier * std_array
    upper_bound = mean_array + std_multiplier * std_array

    # Limit the lower bound to 0
    lower_bound = np.maximum(lower_bound, lower_limit)

    # Limit the upper bound to the specified upper_limit
    upper_bound = np.where(upper_bound > upper_limit, upper_limit, upper_bound)

    return mean_array, std_array, lower_bound, upper_bound

train_rnn_top1acc_mean, train_rnn_top1acc_std, train_rnn_top1acc_lower_bound, train_rnn_top1acc_upper_bound = compute_mean_std(train_rnn_top1acc_runs)
train_grurnn_top1acc_mean, train_grurnn_top1acc_std, train_grurnn_top1acc_lower_bound, train_grurnn_top1acc_upper_bound = compute_mean_std(train_grurnn_top1acc_runs)
train_lstmrnn_top1acc_mean, train_lstmrnn_top1acc_std, train_lstmrnn_top1acc_lower_bound, train_lstmrnn_top1acc_upper_bound = compute_mean_std(train_lstmrnn_top1acc_runs)
train_urlstmrnn_top1acc_mean, train_urlstmrnn_top1acc_std, train_urlstmrnn_top1acc_lower_bound, train_urlstmrnn_top1acc_upper_bound = compute_mean_std(train_urlstmrnn_top1acc_runs)
train_hippo_top1acc_mean, train_hippo_top1acc_std, train_hippo_top1acc_lower_bound, train_hippo_top1acc_upper_bound = compute_mean_std(train_hippo_top1acc_runs)
train_gatedhippornn_top1acc_mean, train_gatedhippornn_top1acc_std, train_gatedhippornn_top1acc_lower_bound, train_gatedhippornn_top1acc_upper_bound = compute_mean_std(train_gatedhippornn_top1acc_runs)


test_rnn_top1acc_mean, test_rnn_top1acc_std, test_rnn_top1acc_lower_bound, test_rnn_top1acc_upper_bound = compute_mean_std(test_rnn_top1acc_runs)
test_grurnn_top1acc_mean, test_grurnn_top1acc_std, test_grurnn_top1acc_lower_bound, test_grurnn_top1acc_upper_bound = compute_mean_std(test_grurnn_top1acc_runs)
test_lstmrnn_top1acc_mean, test_lstmrnn_top1acc_std, test_lstmrnn_top1acc_lower_bound, test_lstmrnn_top1acc_upper_bound = compute_mean_std(test_lstmrnn_top1acc_runs)
test_urlstmrnn_top1acc_mean, test_urlstmrnn_top1acc_std, test_urlstmrnn_top1acc_lower_bound, test_urlstmrnn_top1acc_upper_bound = compute_mean_std(test_urlstmrnn_top1acc_runs)
test_hippo_top1acc_mean, test_hippo_top1acc_std, test_hippo_top1acc_lower_bound, test_hippo_top1acc_upper_bound = compute_mean_std(test_hippo_top1acc_runs)
test_gatedhippornn_top1acc_mean, test_gatedhippornn_top1acc_std, test_gatedhippornn_top1acc_lower_bound, test_gatedhippornn_top1acc_upper_bound = compute_mean_std(test_gatedhippornn_top1acc_runs)


plt.figure(figsize=(23,10))
plt.subplot(1,2,1)
# Plot the mean line
plt.plot(train_rnn_top1acc_mean, label='Rnn')
# Plot the shaded area
plt.fill_between(np.arange(len(train_rnn_top1acc_mean)), train_rnn_top1acc_lower_bound, train_rnn_top1acc_upper_bound,
                 alpha=0.5,)

plt.plot(train_grurnn_top1acc_mean, label='Gru')
# Plot the shaded area
plt.fill_between(np.arange(len(train_grurnn_top1acc_mean)), train_grurnn_top1acc_lower_bound, train_grurnn_top1acc_upper_bound,
                 alpha=0.5,)

plt.plot(train_lstmrnn_top1acc_mean, label='Lstm')
# Plot the shaded area
plt.fill_between(np.arange(len(train_lstmrnn_top1acc_mean)), train_lstmrnn_top1acc_lower_bound, train_lstmrnn_top1acc_upper_bound,
                 alpha=0.5,)

plt.plot(train_urlstmrnn_top1acc_mean, label='UrLstm')
# Plot the shaded area
plt.fill_between(np.arange(len(train_urlstmrnn_top1acc_mean)), train_urlstmrnn_top1acc_lower_bound, train_urlstmrnn_top1acc_upper_bound,
                 alpha=0.5,)

plt.plot(train_hippo_top1acc_mean, label='Hippo')
# Plot the shaded area
plt.fill_between(np.arange(len(train_hippo_top1acc_mean)), train_hippo_top1acc_lower_bound, train_hippo_top1acc_upper_bound,
                 alpha=0.5,)

plt.plot(train_gatedhippornn_top1acc_mean, label='HippoRnn')
# Plot the shaded area
plt.fill_between(np.arange(len(train_gatedhippornn_top1acc_mean)), train_gatedhippornn_top1acc_lower_bound,train_gatedhippornn_top1acc_upper_bound,
                 alpha=0.5,)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training top-1-acc: Mean accuracy and standard deviation.')
plt.legend(loc='center left', bbox_to_anchor=(1, .75), frameon=False)

#plt.figure(figsize=(13,5))
plt.subplot(1,2,2)
# Plot the mean line
plt.plot(test_rnn_top1acc_mean, label='Rnn')
# Plot the shaded area
plt.fill_between(np.arange(len(test_rnn_top1acc_mean)), test_rnn_top1acc_lower_bound, test_rnn_top1acc_upper_bound,
                 alpha=0.5,)

plt.plot(test_grurnn_top1acc_mean, label='Gru')
# Plot the shaded area
plt.fill_between(np.arange(len(test_grurnn_top1acc_mean)), test_grurnn_top1acc_lower_bound, test_grurnn_top1acc_upper_bound,
                 alpha=0.5,)

plt.plot(test_lstmrnn_top1acc_mean, label='Lstm')
# Plot the shaded area
plt.fill_between(np.arange(len(test_lstmrnn_top1acc_mean)), test_lstmrnn_top1acc_lower_bound, test_lstmrnn_top1acc_upper_bound,
                 alpha=0.5,)

plt.plot(test_urlstmrnn_top1acc_mean, label='UrLstm')
# Plot the shaded area
plt.fill_between(np.arange(len(test_urlstmrnn_top1acc_mean)), test_urlstmrnn_top1acc_lower_bound, test_urlstmrnn_top1acc_upper_bound,
                 alpha=0.5,)

plt.plot(test_hippo_top1acc_mean, label='Hippo')
# Plot the shaded area
plt.fill_between(np.arange(len(test_hippo_top1acc_mean)), test_hippo_top1acc_lower_bound, test_hippo_top1acc_upper_bound,
                 alpha=0.5,)

plt.plot(test_gatedhippornn_top1acc_mean, label='HippoRnn')
# Plot the shaded area
plt.fill_between(np.arange(len(test_gatedhippornn_top1acc_mean)), test_gatedhippornn_top1acc_lower_bound, test_gatedhippornn_top1acc_upper_bound,
                 alpha=0.5,)

plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test top-1-acc: Mean accuracy and standard deviation.')
#plt.legend(loc='center left', bbox_to_anchor=(1, .75), frameon=False)
plt.savefig('figures/rnn_train_vs_test_acc.png')