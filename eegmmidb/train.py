import os
import sys
sys.path.append("../utils/")
import torch
import numpy as np
import torch.nn as nn
from torch import Tensor
from typing import List, Tuple
from dataset import ToTensor, EEGDataset2DLeftRight, EEGDatasetRightFeet, EEGDatasetLeftFeet
from utility import train_validate_split_subjects, samples_per_class
from torch.utils.data import Dataset, DataLoader, sampler
import torch.optim as optim
from snn import WrapCUBASpikingCNN


def test_accuracy(network, test_loader, device):
    """
    Return the accuracy of the prediction of the network compared to the ground truth of test data.
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

    return class_correct.sum()/class_total.sum(), class_correct / class_total

def train_network(dataset=EEGDataset2DLeftRight, network=WrapCUBASpikingCNN,
                  dataset_kwargs=dict(), spike_ts=160, param_list=[],
                  validate_subject_list=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                  lr=[0.0001, 0.00001, 2*0.0001], weight_decays=[], batch_size=64, epoch=10):
    """
    Train SNN for EEGMMIDB
    :param dataset: dataset class
    :param network: network class
    :param dataset_kwargs: parameters for dataset class
    :param spike_ts: spike timesteps
    :param param_list: parameters for neuron layers (cdecay, vdecay, vth, grad_win, th_amp, th_decay, base_th)
    :param validate_subject_list: subject list for validation
    :param lr: learning rates (neuron lr, timestep lr, weights lr)
    :param weight_decays: weight decay factors (neuron decay, timestep decay, weights decay)
    :param batch_size: batch size
    :param epoch: number of epochs for training
    :return:
    """

    device = torch.device("cuda")

    # Setup Network
    net = network(spike_ts, device, param_list=param_list)
    net = nn.DataParallel(net.to(device))
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

    decays = ['module.snn.c1_vdecay', 'module.snn.c2_vdecay', 'module.snn.c3_vdecay', 'module.snn.tc1_vdecay', 
    'module.snn.tc1_cdecay', 'module.snn.r1_vdecay', 'module.snn.f1_vdecay', 'module.snn.c1_cdecay', 
    'module.snn.c2_cdecay', 'module.snn.c3_cdecay', 'module.snn.r1_cdecay', 'module.snn.f1_cdecay']
    
    ts_weights = ['module.snn.ts_weights']
    
    decay_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in decays, net.named_parameters()))))

    ts_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in ts_weights, net.named_parameters()))))
    
    weights = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in decays+ts_weights, net.named_parameters()))))


    optimizer = optim.Adam([{'params': weights}, {'params': decay_params, 'lr': lr[0]}, 
    {'params': ts_params, 'lr': lr[1]}], lr=lr[2])

    # Start Training
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
              (class_acc[0]*100, class_acc[1]*100))


if __name__ == "__main__":

    """
    Dataset Parameters
    """

    DATASET = EEGDataset2DLeftRight
    USE_IMAGERY= False
    ds_params = {"base_route": "../utils/eegmmidb_slice_norm/",
                 "subject_id_list":  [i + 1 for i in range(109) if not (i + 1 in [88, 89, 92, 100, 104, 106])],
                 "start_ts": 0,
                 "end_ts": 161,
                 "window_ts": 160,
                 "overlap_ts": 0,
                 "use_imagery": USE_IMAGERY,
                 "transform": ToTensor()}
    """
    Define All Training Parameters
    """
    SPIKE_TS = 160
    BATCH_SIZE = 64
    WT_LR = 0.0001
    TS_LR = 0.0001
    NEURON_LR = 0.0001
    N_CLASSES = 2
    VDECAY = 0.1
    CDECAY = 0.1
    VTH = 0.1
    GRAD_WIN = 0.3
    TH_AMP = 0.01
    TH_DECAY = 0.1
    BASE_TH = 0.01
    WT_DECAY = 1e-6
    TS_DECAY = 2 * WT_DECAY
    NEURON_DECAY = 2 * WT_DECAY
    PARAM_LIST = [CDECAY, VDECAY, VTH, GRAD_WIN, TH_AMP, TH_DECAY, BASE_TH]

    val_list = [[i+1  for i in range(10)],
                [i+11  for i in range(10)],
                [i+21  for i in range(10)],
                [i+31  for i in range(10)],
                [i+41  for i in range(10)],
                [i+51  for i in range(10)],
                [i+61  for i in range(10)],
                [i+71  for i in range(10)],
                [81, 82, 83, 84, 85, 86, 87, 90, 91, 93],
                [94, 95, 96, 97, 98, 99, 101, 102, 103,  105]]

    train_network(dataset_kwargs=ds_params, spike_ts=SPIKE_TS, param_list=PARAM_LIST, dataset=DATASET,
                  batch_size=BATCH_SIZE, epoch=20, lr=[NEURON_LR, TS_LR, WT_LR], weight_decays = [NEURON_DECAY, TS_DECAY, WT_DECAY], 
                  validate_subject_list=val_list[0])
