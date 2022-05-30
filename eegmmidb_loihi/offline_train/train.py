import os
from tokenize import group
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, sampler
import torch.optim as optim
import numpy as np
from dataset import EEGDataset2DLeftRight, ToTensor
from snn_convnet_dropout import WrapCUBASpikingCNN
from train_utils import samples_per_class, train_validate_split_subjects


def test_accuracy(network, test_loader, device):
    """
    Return the accuracy of the prediction of the network compares to the ground truth of test data.
    :param network: Trained Pytorch network
    :param test_loader: Dataloader for test data
    :param device: device
    :return: overall accuracy, class accuracy
    """
    with torch.no_grad():
        class_correct = np.zeros(2)
        class_total = np.zeros(2)
        for data in test_loader:
            eeg_data, label = data
            eeg_data = eeg_data.to(device)
            output = network(eeg_data)
            _, predicted = torch.max(output, 1)
            c = (predicted.to('cpu') == label).numpy()

            for i in range(label.size(0)):
                la = label[i]
                class_correct[la] += c[i]
                class_total[la] += 1.
    return class_correct.sum() / class_total.sum(), class_correct / class_total


def train_network(dataset=EEGDataset2DLeftRight, network=WrapCUBASpikingCNN,
                  dataset_kwargs=dict(), spike_ts=160, param_list=[], c1=64, c2=128, c3=256,
                  validate_subject_list=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                  lr=[0.00001, 2 * 0.0001], batch_size=64, epoch=10, use_cuda=True, save_name='mmidb_conv_group_0'):
    """
    Train Regular DNN for EEGMMIDB
    :param dataset: dataset class
    :param network: network class
    :param dataset_kwargs: parameters for dataset class
    :param validate_subject_list: subject list for validation
    :param batch_size: batch size
    :param epoch: number of epochs for training
    :param use_cuda: if true use cuda
    :param save_name: save name
    :return:
    """
    try:
        os.mkdir("../model/" + save_name)
        print("Directory save_models Created")
    except FileExistsError:
        print("Directory params already exists")
    
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    # Setup Network
    net = network(spike_ts, device, param_list=param_list, c1=c1, c2=c2, c3=c3)
    net.to(device)
    net.train()

    # Setup Dataset and Dataloader
    ds = dataset(**dataset_kwargs)
    train_indices, val_indices = train_validate_split_subjects(ds, validate_subject_list)
    print("Training Samples per Class: ")
    samples_per_class(ds.label[train_indices])
    print("Validate Samples per Class: ")
    samples_per_class(ds.label[val_indices])
    train_sampler = sampler.SubsetRandomSampler(train_indices)
    val_sampler = sampler.SubsetRandomSampler(val_indices)
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                              sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            sampler=val_sampler, num_workers=4, pin_memory=True)

    # Setup Optimizer and loss function

    criterion = nn.CrossEntropyLoss()

    ts_weights = ['module.snn.ts_weights']

    ts_params = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] in ts_weights, net.named_parameters()))))

    weights = list(map(lambda x: x[1], list(filter(lambda kv: kv[0] not in ts_weights, net.named_parameters()))))

    optimizer = optim.Adam([{'params': weights}, {'params': ts_params, 'lr': lr[0]}], lr=lr[1])

    # Start Training
    test_acc_list = np.zeros(epoch)
    for e in range(epoch):
        running_loss = 0
        train_ita = 0
        for i, data in enumerate(train_loader, 0):
            eeg_data, label = data
            eeg_data, label = eeg_data.to(device), label.to(device)
            optimizer.zero_grad()
            output = net(eeg_data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.to('cpu').item()
            train_ita = i
        net.eval()
        acc, class_acc = test_accuracy(net, val_loader, device)
        net.train()

        print('Epoch: %d, Loss: %.3f' % (e, running_loss / (train_ita + 1)))
        print('Overall validation Accuracy after Epoch %d: %.3f %%' % (e, acc * 100))
        print('Accuracy for class 0: %.3f %%, 1: %.3f %%' %
              (class_acc[0] * 100, class_acc[1] * 100))

        net.to('cpu')
        torch.save(net.state_dict(), '../model/' + save_name + '/e' + str(e) + '.pt')
        net.to(device)

        test_acc_list[e] = acc
    
    with open('../model/' + save_name + '/test_accuracy.npy', 'wb+') as f:
        np.save(f, test_acc_list)
    
    print("Finish training ", save_name)


if __name__ == '__main__':
    ds_params = {"base_route": "../data/eegmmidb_slice_norm/",
                 "subject_id_list": [i + 1 for i in range(109) if not (i + 1 in [88, 89, 92, 100, 104, 106])],
                 "start_ts": 0,
                 "end_ts": 161,
                 "window_ts": 160,
                 "overlap_ts": 0,
                 "use_imagery": False,
                 "use_no_movement": False,
                 "transform": ToTensor()}
    """
    Define All Training Parameters
    """
    SPIKE_TS = 160
    BATCH_SIZE = 64
    WT_LR = 1e-4
    TS_LR = 1e-4
    N_CLASSES = 2
    VDECAY = 0.1
    CDECAY = 0.1
    VTH = 0.2
    GRAD_WIN = 0.3
    PARAM_LIST = [CDECAY, VDECAY, VTH, GRAD_WIN]
    C1 = 4
    C2 = 8
    C3 = 128
    
    NET_NAME = 'conv_loihi'
    GROUP = 9
    EPOCH = 100

    # VAD_LIST = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # VAD_LIST = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    # VAD_LIST = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
    # VAD_LIST = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    # VAD_LIST = [40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
    # VAD_LIST = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]
    # VAD_LIST = [60, 61, 62, 63, 64, 65, 66, 67, 68, 69]
    # VAD_LIST = [70, 71, 72, 73, 74, 75, 76, 77, 78, 79]
    # VAD_LIST = [80, 81, 82, 83, 84, 85, 86, 87, 91, 93]
    VAD_LIST = [94, 95, 96, 97, 98, 99, 101, 102, 103, 105, 107, 108, 109]

    train_network(dataset_kwargs=ds_params, spike_ts=SPIKE_TS, param_list=PARAM_LIST, c1=C1, c2=C2, c3=C3,
                  batch_size=BATCH_SIZE, epoch=EPOCH, lr=[TS_LR, WT_LR], validate_subject_list=VAD_LIST,
                  save_name='mmidb_'+ NET_NAME + '_group_' + str(GROUP))
