import torch
import torch.nn as nn
from torch.utils.data import DataLoader, sampler
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import nxsdk.api.n2a as nx
import sys

sys.path.append('../')
from online_loihi_inf.torch_network import WrapCUBASpikingCNN
from offline_train.dataset import EEGDataset2DLeftRight, ToTensor
from offline_train.train_utils import samples_per_class, train_validate_split_subjects
from online_loihi_inf.loihi_network import PyTorch2LoihiNetwork
from online_loihi_inf.loihi_spike_streamer import SpikeStreamer4LoihiNetwork
from online_loihi_inf.loihi_online_input_output import OnlineInputOutput4LoihiNetwork


def build_online_loihi_network(torch_net, spike_ts, extra_ts, param_list, c1, c2, c3):
    """
    Build Loihi network with online encoding and decoding

    :param torch_net: PyTorch network
    :param spike_ts: spike timestep
    :param extra_ts: extra timestep
    :param param_list: neuron parameters
    :param c1: Conv layer 1 dimension
    :param c2: Conv layer 2 dimension
    :param c3: Conv layer 3 dimension
    :return: loihi_board, spike_streamer, fc_decoder_channel
    """
    # Load PyTorch network weights
    raw_c2_weight = torch_net.snn.conv2.psp_func.weight.data.numpy()
    raw_c2_bias = torch_net.snn.conv2.psp_func.bias.data.numpy()
    raw_c3_weight = torch_net.snn.conv3.psp_func.weight.data.numpy()
    raw_c3_weight = PyTorch2LoihiNetwork.combine_conv_layer_weight_w_avgpool(raw_c3_weight)
    raw_c3_bias = torch_net.snn.conv3.psp_func.bias.data.numpy()
    raw_tc1_weight_list = []
    raw_tc1_bias_list = []
    for ww in range(3):
        raw_tc1_weight_list.append(torch_net.snn.temp_conv1.psp_func_list[ww].weight.data.numpy())
        raw_tc1_bias_list.append(torch_net.snn.temp_conv1.psp_func_list[ww].bias.data.numpy())
    raw_r1_weight = torch_net.snn.rec1.rec_func.weight.data.numpy()
    raw_r1_bias = torch_net.snn.rec1.rec_func.bias.data.numpy()
    raw_f1_weight = torch_net.snn.fc1.psp_func.weight.data.numpy()
    raw_f1_bias = torch_net.snn.fc1.psp_func.bias.data.numpy()

    # Build Loihi network
    loihi_core_resource = [{'compartment': 14, 'fanin': 16384, 'fanout': 16384} for _ in range(256)]
    loihi_net = nx.NxNet()
    neuron_spec = {'cdecay': param_list[0], 'vdecay': param_list[1], 'vth': param_list[2]}
    # Create FC layer first for online decoding
    f1_layer_spec = {'raw_weight': raw_f1_weight, 'raw_bias': raw_f1_bias, 'dimension': 128}
    f1_layer, f1_w_b_dict = OnlineInputOutput4LoihiNetwork.online_fc_layer(loihi_net, neuron_spec, f1_layer_spec,
                                                                           loihi_core_resource, c3, 1)
    OnlineInputOutput4LoihiNetwork.online_fc_decoding(f1_layer)
    # Create input layer and list of bias layers
    input_layer, spike_streamer = SpikeStreamer4LoihiNetwork.spike_streamer_2d_layer(loihi_net,
                                                                                     {'size_x': 9, 'size_y': 8,
                                                                                      'size_c': c1},
                                                                                     loihi_core_resource,
                                                                                     1)
    bias_layer_list, pseudo_2_bias_list = OnlineInputOutput4LoihiNetwork.online_bias_layers(loihi_net, 7,
                                                                                            loihi_core_resource,
                                                                                            [c2 * 6 * 7, c3, c3, c3,
                                                                                             c3, c3, c3])
    # Create Conv2 layer
    c2_layer_spec = {'raw_weight': raw_c2_weight, 'raw_bias': raw_c2_bias,
                     'size_x': 7, 'size_y': 6, 'size_c': c2,
                     'conv_x': 3, 'conv_y': 3, 'conv_c': c1,
                     'stride_x': 1, 'stride_y': 1}
    c2_layer = PyTorch2LoihiNetwork.conv_layer(input_layer, bias_layer_list[0],
                                               neuron_spec, c2_layer_spec,
                                               loihi_core_resource, c1 * 9, c3 * 36)
    # Create Conv3 layer
    c3_layer_spec = {'raw_weight': raw_c3_weight, 'raw_bias': raw_c3_bias,
                     'size_x': 1, 'size_y': 1, 'size_c': c3,
                     'conv_x': 6, 'conv_y': 6, 'conv_c': c2,
                     'stride_x': 2, 'stride_y': 2}
    c3_layer = PyTorch2LoihiNetwork.conv_layer(c2_layer, bias_layer_list[1],
                                               neuron_spec, c3_layer_spec,
                                               loihi_core_resource, c2 * 36, c3 * 3)
    # Create Temp Conv layer
    tc1_layer_spec = {'raw_weight_list': raw_tc1_weight_list, 'raw_bias_list': raw_tc1_bias_list,
                      'window': 3, 'dimension': c3}
    tc1_layer = PyTorch2LoihiNetwork.temp_conv_layer(c3_layer,
                                                     [bias_layer_list[2],
                                                      bias_layer_list[3],
                                                      bias_layer_list[4]],
                                                     neuron_spec, tc1_layer_spec,
                                                     loihi_core_resource, c3 * 3, 1)
    # Create Recurrent layer
    r1_layer_spec = {'raw_rec_weight': raw_r1_weight, 'raw_rec_bias': raw_r1_bias, 'dimension': c3}
    r1_layer = PyTorch2LoihiNetwork.recurrent_layer_w_identity_input(tc1_layer, bias_layer_list[5],
                                                                     neuron_spec, r1_layer_spec,
                                                                     loihi_core_resource, c3 + 1, c3 + 128)
    # Connect Recurrent layer with online FC layer
    OnlineInputOutput4LoihiNetwork.online_fc_layer_connection(r1_layer, bias_layer_list[6], f1_layer, f1_w_b_dict)

    # compile the board
    compiler = nx.N2Compiler()
    loihi_board = compiler.compile(loihi_net)
    # Setup spike streamer and decoder
    spike_streamer.configureStreamer(loihi_board)
    bias_spike_info_list = [[1, spike_ts + 1],
                            [2, spike_ts + 2],
                            [5, spike_ts + 5],
                            [4, spike_ts + 4],
                            [3, spike_ts + 3],
                            [4, spike_ts + 4],
                            [5, spike_ts + 5]]
    OnlineInputOutput4LoihiNetwork.setup_bias_encoding_snip('./snips', loihi_board, loihi_net, pseudo_2_bias_list,
                                                            bias_spike_info_list, spike_ts, extra_ts)
    ts_weights = torch_net.snn.ts_weights.data.numpy().squeeze()
    fc_decoder_channel = OnlineInputOutput4LoihiNetwork.setup_fc_decoding_snip('./snips', loihi_board, 128, ts_weights,
                                                                               6, spike_ts + 5,
                                                                               spike_ts, extra_ts)

    return loihi_board, spike_streamer, fc_decoder_channel


def test_online_loihi_network(model_dir, dataset=EEGDataset2DLeftRight, network=WrapCUBASpikingCNN,
                              dataset_kwargs=dict(), spike_ts=160, extra_ts=7, param_list=[], c1=4, c2=8, c3=128,
                              validate_subject_list=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19], test_num=10):
    """
    Test online loihi network

    :param model_dir: saved model directory
    :param dataset: dataset for testing
    :param network: PyTorch network
    :param dataset_kwargs: dataset arguments
    :param spike_ts: spike timesteps
    :param extra_ts: extra timesteps
    :param param_list: neuron parameters
    :param c1: Conv layer 1 dimension
    :param c2: Conv layer 2 dimension
    :param c3: Conv layer 3 dimension
    :param validate_subject_list: list of validation subjects
    :param test_num: number of test cases
    """
    # Setup Dataset and Dataloader
    ds = dataset(**dataset_kwargs)
    train_indices, val_indices = train_validate_split_subjects(ds, validate_subject_list)
    print("Validate Samples per Class: ")
    samples_per_class(ds.label[val_indices])
    val_sampler = sampler.SubsetRandomSampler(val_indices)
    val_loader = DataLoader(ds, batch_size=1, shuffle=False, sampler=val_sampler)

    # Load PyTorch network and get trained weights and biases
    device = torch.device("cpu")
    torch_net = network(spike_ts, device, param_list=param_list, c1=c1, c2=c2, c3=c3)
    torch_net.load_state_dict(torch.load(model_dir))

    # Create SNN
    loihi_board, spike_streamer, fc_channel = build_online_loihi_network(torch_net, spike_ts, extra_ts, param_list,
                                                                         c1, c2, c3)
    # Start running SNN
    loihi_board.run((spike_ts + extra_ts) * test_num, aSync=True)

    with torch.no_grad():
        loihi_class_correct = np.zeros(2)
        torch_class_correct = np.zeros(2)
        class_total = np.zeros(2)
        for idx, data in enumerate(val_loader, 0):
            eeg_data, label = data
            # Encode spike and add to Loihi
            c1_spikes = torch_net.forward_encoding(eeg_data)
            c1_spikes = c1_spikes.view(-1, 9 * 8 * c1).permute(1, 0).numpy().astype(int)
            SpikeStreamer4LoihiNetwork.add_input_spike_time(spike_streamer, c1_spikes, idx * (spike_ts + extra_ts))
            spike_streamer.advanceTime((idx + 1) * (spike_ts + extra_ts))
            # Read FC1 output and decode
            loihi_weighted_fc1_output = fc_channel.read(128)
            loihi_weighted_fc1_output = np.array(loihi_weighted_fc1_output) / 10000.
            fc2_weight = torch_net.snn.fc2.weight.data.numpy()
            loihi_output = np.matmul(fc2_weight, loihi_weighted_fc1_output).squeeze()
            loihi_label = np.argmax(loihi_output)
            if loihi_label == label[0]:
                loihi_class_correct[loihi_label] += 1
            # Compare
            fc2_outputs = torch_net(eeg_data)
            fc2_outputs = fc2_outputs.numpy().squeeze()
            torch_label = np.argmax(fc2_outputs)
            if torch_label == label[0]:
                torch_class_correct[torch_label] += 1
            class_total[label[0]] += 1
            print(idx)
            if idx + 1 == test_num:
                break

    loihi_board.finishRun()
    loihi_board.disconnect()

    print("Loihi Test Accuracy: ", loihi_class_correct.sum() / class_total.sum(),
          "Loihi Class Accuracy: ", loihi_class_correct / class_total,
          "PyTorch Test Accuracy: ", torch_class_correct.sum() / class_total.sum(),
          "PyTorch Class Accuracy: ", torch_class_correct / class_total)


def find_early_stop_epoch(accuracy_dir, warm_up_epoch=50):
    """
    Find the index of epoch that will do early stop (epoch with max test accraucy after warm up epoch)

    :param accuracy_dir: directory of the accuracy list
    :param warm_up_epoch: warm up epoch
    """
    with open(accuracy_dir, 'rb') as f:
        acc_list = np.load(f)

    stop_idx = np.argmax(acc_list[warm_up_epoch:]) + warm_up_epoch
    stop_acc = acc_list[stop_idx]

    print("Early stop epoch: ", stop_idx, " accuracy: ", stop_acc)
    
    return stop_idx


if __name__ == '__main__':
    ds_params = {"base_route": "../../data/eegmmidb_slice_norm/",
                 "subject_id_list": [i + 1 for i in range(109) if not (i + 1 in [88, 89, 92, 100, 104, 106])],
                 "start_ts": 0,
                 "end_ts": 161,
                 "window_ts": 160,
                 "overlap_ts": 0,
                 "use_imagery": False,
                 "use_no_movement": False,
                 "transform": ToTensor()}
    """
    Define All Testing Parameters
    """
    MODEL_ACC_DIR = '../model/mmidb_conv_loihi_group_9/test_accuracy.npy'
    early_stop_epoch = find_early_stop_epoch(MODEL_ACC_DIR, warm_up_epoch=50)

    MODEL_DIR = '../model/mmidb_conv_loihi_group_9/e' + str(early_stop_epoch) + '.pt'
    SPIKE_TS = 160
    EXTRA_TS = 6 + 20
    BATCH_SIZE = 1
    VDECAY = 0.1
    CDECAY = 0.1
    VTH = 0.2
    GRAD_WIN = 0.3
    PARAM_LIST = [CDECAY, VDECAY, VTH, GRAD_WIN]
    C1 = 4
    C2 = 8
    C3 = 128
    # TEST_NUM = 450
    TEST_NUM = 585

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

    test_online_loihi_network(MODEL_DIR, dataset_kwargs=ds_params, spike_ts=SPIKE_TS, extra_ts=EXTRA_TS,
                              param_list=PARAM_LIST,
                              c1=C1, c2=C2, c3=C3,
                              validate_subject_list=VAD_LIST,
                              test_num=TEST_NUM)
