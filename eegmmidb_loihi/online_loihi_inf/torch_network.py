import torch
import numpy as np
import torch.nn as nn
import sys

sys.path.append('../')
from offline_train.snn import PseudoSpikeRect, FeedForwardCUBALIFCell, RecurrentCUBALIFCell, TemporalConvCUBALIFCell


class CUBASpikingCNN(nn.Module):
    def __init__(self, spike_ts, params, c1, c2, c3, device):
        """
        :param spike_ts: spike timesteps
        :param params: list of param for each neuron layer
        :param c1: channel dimension for 1st conv layer
        :param c2: channel dimension for 2nd conv layer
        :param c3: channel dimension for 3rd conv layer
        :param device: device
        """
        super(CUBASpikingCNN, self).__init__()
        self.spike_ts = spike_ts
        self.cdecay, self.vdecay, self.vth, self.grad_win = params
        pseudo_grad_ops = PseudoSpikeRect.apply

        self.ts_weights = nn.Parameter(torch.ones((self.spike_ts, 1), device=device) / self.spike_ts)

        self.conv1 = FeedForwardCUBALIFCell(nn.Conv2d(1, c1, (3, 3), bias=True),
                                            pseudo_grad_ops,
                                            params)
        self.conv2 = FeedForwardCUBALIFCell(nn.Conv2d(c1, c2, (3, 3), bias=True),
                                            pseudo_grad_ops,
                                            params)
        self.avg_pool = nn.AvgPool2d(2)
        self.conv3 = FeedForwardCUBALIFCell(nn.Conv2d(c2, c3, (3, 3), bias=True),
                                            pseudo_grad_ops,
                                            params)
        self.temp_conv1 = TemporalConvCUBALIFCell(3,
                                                  [nn.Linear(c3, c3, bias=True) for _ in range(3)],
                                                  pseudo_grad_ops,
                                                  params)
        self.rec1 = RecurrentCUBALIFCell(nn.Identity(),
                                         nn.Linear(c3, c3, bias=True),
                                         pseudo_grad_ops,
                                         params)
        self.fc1 = FeedForwardCUBALIFCell(nn.Linear(c3, 128, bias=True),
                                          pseudo_grad_ops,
                                          params)
        self.fc2 = nn.Linear(128, 2, bias=False)

    def forward(self, input_data, states):
        """
        :param input_data: input EEG raw data
        :param states: list of (init spike, init voltage)
        :return: output
        """
        batch_size = input_data.shape[0]
        output_spikes = []
        temp_conv_spike_buffer = []
        c1_state, c2_state, c3_state, tc1_state, r1_state, f1_state = \
            states[0], states[1], states[2], states[3], states[4], states[5]

        for step in range(self.spike_ts):
            input_signal = input_data[:, :, :, :, step]
            c1_spike, c1_state = self.conv1(input_signal, c1_state)
            c2_spike, c2_state = self.conv2(c1_spike, c2_state)
            avg_pool_c2_spike = self.avg_pool(c2_spike)
            c3_spike, c3_state = self.conv3(avg_pool_c2_spike, c3_state)
            flat_c3_spike = c3_spike.view(batch_size, -1)

            temp_conv_spike_buffer.append(flat_c3_spike)

            if len(temp_conv_spike_buffer) > 3:
                temp_conv_spike_buffer.pop(0)

            tc1_spike, tc1_state = self.temp_conv1(temp_conv_spike_buffer, tc1_state)

            r1_spike, r1_state = self.rec1(tc1_spike, r1_state)

            f1_spike, f1_state = self.fc1(r1_spike, f1_state)
            f2_output = self.fc2(f1_spike)
            output_spikes += [f2_output * self.ts_weights[step]]
        outputs = torch.stack(output_spikes).sum(dim=0)

        return outputs

    def forward_encoding(self, input_data, states):
        """
        Encoding Spike Inputs (c1) to Loihi Network

        :param input_data: input EEG raw data
        :param states: list of (init spike, init voltage)
        :return: output_spike
        """
        output_spike = []
        c1_state = states[0]
        for step in range(self.spike_ts):
            input_signal = input_data[:, :, :, :, step]
            c1_spike, c1_state = self.conv1(input_signal, c1_state)
            output_spike.append(c1_spike)
        output_spike = torch.stack(output_spike)
        return output_spike

    def forward_decoding(self, f1_spike):
        """
        Decode Loihi Network Prediction to class prediction

        :param f1_spike: output spikes of F1 layer from Loihi network
        :return: output
        """
        output_spikes = []
        for step in range(self.spike_ts):
            f2_output = self.fc2(f1_spike[:, :, step])
            output_spikes += [f2_output * self.ts_weights[step]]
        output = torch.stack(output_spikes).sum(dim=0)
        return output

    def forward_ablation_c2(self, input_data, states):
        """
        Output Spike Layer C2 for network ablation testing

        :param input_data: input EEG raw data
        :param states: list of (init spike, init voltage)
        :return: output_spike
        """
        output_spike = []
        c1_state, c2_state = states[0], states[1]
        for step in range(self.spike_ts):
            input_signal = input_data[:, :, :, :, step]
            c1_spike, c1_state = self.conv1(input_signal, c1_state)
            c2_spike, c2_state = self.conv2(c1_spike, c2_state)
            output_spike.append(c2_spike)
        output_spike = torch.stack(output_spike)
        return output_spike

    def forward_ablation_c3(self, input_data, states):
        """
        Output Spike Layer C3 for network ablation testing

        :param input_data: input EEG raw data
        :param states: list of (init spike, init voltage)
        :return: output_spike
        """
        output_spike = []
        c1_state, c2_state, c3_state = states[0], states[1], states[2]
        for step in range(self.spike_ts):
            input_signal = input_data[:, :, :, :, step]
            c1_spike, c1_state = self.conv1(input_signal, c1_state)
            c2_spike, c2_state = self.conv2(c1_spike, c2_state)
            avg_pool_c2_spike = self.avg_pool(c2_spike)
            c3_spike, c3_state = self.conv3(avg_pool_c2_spike, c3_state)
            output_spike.append(c3_spike)
        output_spike = torch.stack(output_spike)
        return output_spike

    def forward_ablation_tc1(self, input_data, states):
        """
        Output Spike Layer TC1 for network ablation testing

        :param input_data: input EEG raw data
        :param states: list of (init spike, init voltage)
        :return: output_spike
        """
        batch_size = input_data.shape[0]
        output_spike = []
        temp_conv_spike_buffer = []
        c1_state, c2_state, c3_state, tc1_state = \
            states[0], states[1], states[2], states[3]

        for step in range(self.spike_ts):
            input_signal = input_data[:, :, :, :, step]
            c1_spike, c1_state = self.conv1(input_signal, c1_state)
            c2_spike, c2_state = self.conv2(c1_spike, c2_state)
            avg_pool_c2_spike = self.avg_pool(c2_spike)
            c3_spike, c3_state = self.conv3(avg_pool_c2_spike, c3_state)
            flat_c3_spike = c3_spike.view(batch_size, -1)

            temp_conv_spike_buffer.append(flat_c3_spike)

            if len(temp_conv_spike_buffer) > 3:
                temp_conv_spike_buffer.pop(0)

            tc1_spike, tc1_state = self.temp_conv1(temp_conv_spike_buffer, tc1_state)
            output_spike.append(tc1_spike)
        output_spike = torch.stack(output_spike)
        return output_spike

    def forward_ablation_r1(self, input_data, states):
        """
        Output Spike Layer R1 for network ablation testing

        :param input_data: input EEG raw data
        :param states: list of (init spike, init voltage)
        :return: output_spike
        """
        batch_size = input_data.shape[0]
        output_spike = []
        temp_conv_spike_buffer = []
        c1_state, c2_state, c3_state, tc1_state, r1_state = \
            states[0], states[1], states[2], states[3], states[4]

        for step in range(self.spike_ts):
            input_signal = input_data[:, :, :, :, step]
            c1_spike, c1_state = self.conv1(input_signal, c1_state)
            c2_spike, c2_state = self.conv2(c1_spike, c2_state)
            avg_pool_c2_spike = self.avg_pool(c2_spike)
            c3_spike, c3_state = self.conv3(avg_pool_c2_spike, c3_state)
            flat_c3_spike = c3_spike.view(batch_size, -1)

            temp_conv_spike_buffer.append(flat_c3_spike)

            if len(temp_conv_spike_buffer) > 3:
                temp_conv_spike_buffer.pop(0)

            tc1_spike, tc1_state = self.temp_conv1(temp_conv_spike_buffer, tc1_state)

            r1_spike, r1_state = self.rec1(tc1_spike, r1_state)
            output_spike.append(r1_spike)
        output_spike = torch.stack(output_spike)
        return output_spike

    def forward_ablation_f1(self, input_data, states):
        """
        Output Spike Layer F1 for network ablation testing

        :param input_data: input EEG raw data
        :param states: list of (init spike, init voltage)
        :return: output_spike
        """
        batch_size = input_data.shape[0]
        output_spike = []
        temp_conv_spike_buffer = []
        c1_state, c2_state, c3_state, tc1_state, r1_state, f1_state = \
            states[0], states[1], states[2], states[3], states[4], states[5]

        for step in range(self.spike_ts):
            input_signal = input_data[:, :, :, :, step]
            c1_spike, c1_state = self.conv1(input_signal, c1_state)
            c2_spike, c2_state = self.conv2(c1_spike, c2_state)
            avg_pool_c2_spike = self.avg_pool(c2_spike)
            c3_spike, c3_state = self.conv3(avg_pool_c2_spike, c3_state)
            flat_c3_spike = c3_spike.view(batch_size, -1)

            temp_conv_spike_buffer.append(flat_c3_spike)

            if len(temp_conv_spike_buffer) > 3:
                temp_conv_spike_buffer.pop(0)

            tc1_spike, tc1_state = self.temp_conv1(temp_conv_spike_buffer, tc1_state)

            r1_spike, r1_state = self.rec1(tc1_spike, r1_state)

            f1_spike, f1_state = self.fc1(r1_spike, f1_state)
            output_spike.append(f1_spike)
        output_spike = torch.stack(output_spike)
        return output_spike


class WrapCUBASpikingCNN(nn.Module):
    def __init__(self, spike_ts, device, param_list, c1=64, c2=128, c3=256):
        """
        :param spike_ts: spike timesteps
        :param device: device
        :param param_list: list of param for each neuron layer
        :param c1: channel dimension for 1st conv layer
        :param c2: channel dimension for 2nd conv layer
        :param c3: channel dimension for 3rd conv layer
        """
        super(WrapCUBASpikingCNN, self).__init__()
        self.device = device
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.snn = CUBASpikingCNN(spike_ts, param_list, c1, c2, c3, device)

    def forward(self, input_data):
        """
        :param input_data: input EEG spike trains
        :return: output
        """
        batch_size = input_data.shape[0]
        c1_current = torch.zeros(batch_size, self.c1, 8, 9, device=self.device)
        c1_volt = torch.zeros(batch_size, self.c1, 8, 9, device=self.device)
        c1_spike = torch.zeros(batch_size, self.c1, 8, 9, device=self.device)
        c1_state = (c1_spike, c1_current, c1_volt)
        c2_current = torch.zeros(batch_size, self.c2, 6, 7, device=self.device)
        c2_volt = torch.zeros(batch_size, self.c2, 6, 7, device=self.device)
        c2_spike = torch.zeros(batch_size, self.c2, 6, 7, device=self.device)
        c2_state = (c2_spike, c2_current, c2_volt)
        c3_current = torch.zeros(batch_size, self.c3, 1, 1, device=self.device)
        c3_volt = torch.zeros(batch_size, self.c3, 1, 1, device=self.device)
        c3_spike = torch.zeros(batch_size, self.c3, 1, 1, device=self.device)
        c3_state = (c3_spike, c3_current, c3_volt)
        tc1_current = torch.zeros(batch_size, self.c3, device=self.device)
        tc1_volt = torch.zeros(batch_size, self.c3, device=self.device)
        tc1_spike = torch.zeros(batch_size, self.c3, device=self.device)
        tc1_state = (tc1_spike, tc1_current, tc1_volt)
        r1_current = torch.zeros(batch_size, self.c3, device=self.device)
        r1_volt = torch.zeros(batch_size, self.c3, device=self.device)
        r1_spike = torch.zeros(batch_size, self.c3, device=self.device)
        r1_state = (r1_spike, r1_current, r1_volt)
        f1_current = torch.zeros(batch_size, 128, device=self.device)
        f1_volt = torch.zeros(batch_size, 128, device=self.device)
        f1_spike = torch.zeros(batch_size, 128, device=self.device)
        f1_state = (f1_spike, f1_current, f1_volt)
        states = (c1_state, c2_state, c3_state, tc1_state, r1_state, f1_state)
        output = self.snn(input_data, states)
        return output

    def forward_encoding(self, input_data):
        """
        Encoding Spike Inputs (c1) to Loihi Network

        :param input_data: input EEG raw data
        :return: output_spike
        """
        batch_size = input_data.shape[0]
        c1_current = torch.zeros(batch_size, self.c1, 8, 9, device=self.device)
        c1_volt = torch.zeros(batch_size, self.c1, 8, 9, device=self.device)
        c1_spike = torch.zeros(batch_size, self.c1, 8, 9, device=self.device)
        c1_state = (c1_spike, c1_current, c1_volt)
        states = [c1_state]
        output_spike = self.snn.forward_encoding(input_data, states)
        return output_spike

    def forward_decoding(self, f1_spike):
        """
        Decode Loihi Network Prediction to class prediction

        :param f1_spike: output spikes of F1 layer from Loihi network
        :return: output
        """
        output = self.snn.forward_decoding(f1_spike)
        return output

    def forward_ablation_c2(self, input_data):
        """
        Output Spike Layer C2 for network ablation testing

        :param input_data: input EEG raw data
        :return: output_spike
        """
        batch_size = input_data.shape[0]
        c1_current = torch.zeros(batch_size, self.c1, 8, 9, device=self.device)
        c1_volt = torch.zeros(batch_size, self.c1, 8, 9, device=self.device)
        c1_spike = torch.zeros(batch_size, self.c1, 8, 9, device=self.device)
        c1_state = (c1_spike, c1_current, c1_volt)
        c2_current = torch.zeros(batch_size, self.c2, 6, 7, device=self.device)
        c2_volt = torch.zeros(batch_size, self.c2, 6, 7, device=self.device)
        c2_spike = torch.zeros(batch_size, self.c2, 6, 7, device=self.device)
        c2_state = (c2_spike, c2_current, c2_volt)
        states = [c1_state, c2_state]
        output_spike = self.snn.forward_ablation_c2(input_data, states)
        return output_spike

    def forward_ablation_c3(self, input_data):
        """
        Output Spike Layer C3 for network ablation testing

        :param input_data: input EEG raw data
        :return: output_spike
        """
        batch_size = input_data.shape[0]
        c1_current = torch.zeros(batch_size, self.c1, 8, 9, device=self.device)
        c1_volt = torch.zeros(batch_size, self.c1, 8, 9, device=self.device)
        c1_spike = torch.zeros(batch_size, self.c1, 8, 9, device=self.device)
        c1_state = (c1_spike, c1_current, c1_volt)
        c2_current = torch.zeros(batch_size, self.c2, 6, 7, device=self.device)
        c2_volt = torch.zeros(batch_size, self.c2, 6, 7, device=self.device)
        c2_spike = torch.zeros(batch_size, self.c2, 6, 7, device=self.device)
        c2_state = (c2_spike, c2_current, c2_volt)
        c3_current = torch.zeros(batch_size, self.c3, 1, 1, device=self.device)
        c3_volt = torch.zeros(batch_size, self.c3, 1, 1, device=self.device)
        c3_spike = torch.zeros(batch_size, self.c3, 1, 1, device=self.device)
        c3_state = (c3_spike, c3_current, c3_volt)
        states = [c1_state, c2_state, c3_state]
        output_spike = self.snn.forward_ablation_c3(input_data, states)
        return output_spike

    def forward_ablation_tc1(self, input_data):
        """
        Output Spike Layer TC1 for network ablation testing

        :param input_data: input EEG raw data
        :return: output_spike
        """
        batch_size = input_data.shape[0]
        c1_current = torch.zeros(batch_size, self.c1, 8, 9, device=self.device)
        c1_volt = torch.zeros(batch_size, self.c1, 8, 9, device=self.device)
        c1_spike = torch.zeros(batch_size, self.c1, 8, 9, device=self.device)
        c1_state = (c1_spike, c1_current, c1_volt)
        c2_current = torch.zeros(batch_size, self.c2, 6, 7, device=self.device)
        c2_volt = torch.zeros(batch_size, self.c2, 6, 7, device=self.device)
        c2_spike = torch.zeros(batch_size, self.c2, 6, 7, device=self.device)
        c2_state = (c2_spike, c2_current, c2_volt)
        c3_current = torch.zeros(batch_size, self.c3, 1, 1, device=self.device)
        c3_volt = torch.zeros(batch_size, self.c3, 1, 1, device=self.device)
        c3_spike = torch.zeros(batch_size, self.c3, 1, 1, device=self.device)
        c3_state = (c3_spike, c3_current, c3_volt)
        tc1_current = torch.zeros(batch_size, self.c3, device=self.device)
        tc1_volt = torch.zeros(batch_size, self.c3, device=self.device)
        tc1_spike = torch.zeros(batch_size, self.c3, device=self.device)
        tc1_state = (tc1_spike, tc1_current, tc1_volt)
        states = [c1_state, c2_state, c3_state, tc1_state]
        output_spike = self.snn.forward_ablation_tc1(input_data, states)
        return output_spike

    def forward_ablation_r1(self, input_data):
        """
        Output Spike Layer R1 for network ablation testing

        :param input_data: input EEG raw data
        :return: output_spike
        """
        batch_size = input_data.shape[0]
        c1_current = torch.zeros(batch_size, self.c1, 8, 9, device=self.device)
        c1_volt = torch.zeros(batch_size, self.c1, 8, 9, device=self.device)
        c1_spike = torch.zeros(batch_size, self.c1, 8, 9, device=self.device)
        c1_state = (c1_spike, c1_current, c1_volt)
        c2_current = torch.zeros(batch_size, self.c2, 6, 7, device=self.device)
        c2_volt = torch.zeros(batch_size, self.c2, 6, 7, device=self.device)
        c2_spike = torch.zeros(batch_size, self.c2, 6, 7, device=self.device)
        c2_state = (c2_spike, c2_current, c2_volt)
        c3_current = torch.zeros(batch_size, self.c3, 1, 1, device=self.device)
        c3_volt = torch.zeros(batch_size, self.c3, 1, 1, device=self.device)
        c3_spike = torch.zeros(batch_size, self.c3, 1, 1, device=self.device)
        c3_state = (c3_spike, c3_current, c3_volt)
        tc1_current = torch.zeros(batch_size, self.c3, device=self.device)
        tc1_volt = torch.zeros(batch_size, self.c3, device=self.device)
        tc1_spike = torch.zeros(batch_size, self.c3, device=self.device)
        tc1_state = (tc1_spike, tc1_current, tc1_volt)
        r1_current = torch.zeros(batch_size, self.c3, device=self.device)
        r1_volt = torch.zeros(batch_size, self.c3, device=self.device)
        r1_spike = torch.zeros(batch_size, self.c3, device=self.device)
        r1_state = (r1_spike, r1_current, r1_volt)
        states = [c1_state, c2_state, c3_state, tc1_state, r1_state]
        output_spike = self.snn.forward_ablation_r1(input_data, states)
        return output_spike

    def forward_ablation_f1(self, input_data):
        """
        Output Spike Layer F1 for network ablation testing

        :param input_data: input EEG raw data
        :return: output_spike
        """
        batch_size = input_data.shape[0]
        c1_current = torch.zeros(batch_size, self.c1, 8, 9, device=self.device)
        c1_volt = torch.zeros(batch_size, self.c1, 8, 9, device=self.device)
        c1_spike = torch.zeros(batch_size, self.c1, 8, 9, device=self.device)
        c1_state = (c1_spike, c1_current, c1_volt)
        c2_current = torch.zeros(batch_size, self.c2, 6, 7, device=self.device)
        c2_volt = torch.zeros(batch_size, self.c2, 6, 7, device=self.device)
        c2_spike = torch.zeros(batch_size, self.c2, 6, 7, device=self.device)
        c2_state = (c2_spike, c2_current, c2_volt)
        c3_current = torch.zeros(batch_size, self.c3, 1, 1, device=self.device)
        c3_volt = torch.zeros(batch_size, self.c3, 1, 1, device=self.device)
        c3_spike = torch.zeros(batch_size, self.c3, 1, 1, device=self.device)
        c3_state = (c3_spike, c3_current, c3_volt)
        tc1_current = torch.zeros(batch_size, self.c3, device=self.device)
        tc1_volt = torch.zeros(batch_size, self.c3, device=self.device)
        tc1_spike = torch.zeros(batch_size, self.c3, device=self.device)
        tc1_state = (tc1_spike, tc1_current, tc1_volt)
        r1_current = torch.zeros(batch_size, self.c3, device=self.device)
        r1_volt = torch.zeros(batch_size, self.c3, device=self.device)
        r1_spike = torch.zeros(batch_size, self.c3, device=self.device)
        r1_state = (r1_spike, r1_current, r1_volt)
        f1_current = torch.zeros(batch_size, 128, device=self.device)
        f1_volt = torch.zeros(batch_size, 128, device=self.device)
        f1_spike = torch.zeros(batch_size, 128, device=self.device)
        f1_state = (f1_spike, f1_current, f1_volt)
        states = (c1_state, c2_state, c3_state, tc1_state, r1_state, f1_state)
        output_spike = self.snn.forward_ablation_f1(input_data, states)
        return output_spike
