import torch
import numpy as np
import torch.nn as nn
from torch.distributions.bernoulli import Bernoulli


AMP = 0.3
DROPOUT_FC = 0.5

# Define custom autograd function for Spike Function
class PseudoSpikeRect(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, vth, grad_win):
        ctx.save_for_backward(input)
        ctx.vth = vth
        ctx.grad_win = grad_win
        return input.gt(vth).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        vth = ctx.vth
        grad_win = ctx.grad_win
        grad_input = grad_output.clone()
        spike_pseudo_grad = abs(input - vth) < grad_win
        return AMP * grad_input * spike_pseudo_grad.float(), None, None

class PseudoSpikeRectDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, vth, grad_win, mask):
        ctx.save_for_backward(input)
        ctx.vth = vth
        ctx.grad_win = grad_win
        ctx.mask = mask
        return input.gt(vth).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        vth = ctx.vth
        grad_win = ctx.grad_win
        mask = ctx.mask
        grad_input = grad_output.clone()
        spike_pseudo_grad = abs(input - vth) < grad_win
        spike_pseudo_grad[mask==0] = 0
        return AMP * grad_input * spike_pseudo_grad.float(), None, None, None


class FeedForwardCUBALIFCell(nn.Module):
    def __init__(self, psp_func, pseudo_grad_ops, param):
        """
        :param psp_func: pre-synaptic function
        :param pseudo_grad_ops: pseudo gradient operation
        :param param: (current decay, voltage decay, voltage threshold, gradient parameter)
        """
        super(FeedForwardCUBALIFCell, self).__init__()
        self.psp_func = psp_func
        self.pseudo_grad_ops = pseudo_grad_ops
        self.cdecay, self.vdecay, self.vth, self.grad_win = param

    def forward(self, input_data, state):
        """
        :param input_data: input spike from pre-synaptic neurons
        :param state: (output spike of last timestep, current of last timestep, voltage of last timestep)
        :return: output spike, (output spike, current, voltage)
        """
        pre_spike, pre_current, pre_volt = state
        current = self.cdecay * pre_current + self.psp_func(input_data)
        volt = self.vdecay * pre_volt * (1. - pre_spike) + current
        output = self.pseudo_grad_ops(volt, self.vth, self.grad_win)
        return output, (output, current, volt)

class FeedForwardCUBALIFCellDropout(nn.Module):
    def __init__(self, psp_func, pseudo_grad_ops, param):
        """
        :param psp_func: pre-synaptic function
        :param pseudo_grad_ops: pseudo gradient operation
        :param param: (current decay, voltage decay, voltage threshold, gradient parameter)
        """
        super(FeedForwardCUBALIFCellDropout, self).__init__()
        self.psp_func = psp_func
        self.pseudo_grad_ops = pseudo_grad_ops
        self.cdecay, self.vdecay, self.vth, self.grad_win = param

    def forward(self, input_data, state, mask, train):
        """
        :param input_data: input spike from pre-synaptic neurons
        :param state: (output spike of last timestep, current of last timestep, voltage of last timestep)
        :return: output spike, (output spike, current, voltage)
        """
        pre_spike, pre_current, pre_volt = state
        current = self.cdecay * pre_current + self.psp_func(input_data)
        if train is True:
            current = current * mask
        volt = self.vdecay * pre_volt * (1. - pre_spike) + current
        output = self.pseudo_grad_ops(volt, self.vth, self.grad_win, mask)
        return output, (output, current, volt)


class RecurrentCUBALIFCell(nn.Module):
    def __init__(self, psp_func, rec_func, pseudo_grad_ops, param):
        """
        :param psp_func: pre-synaptic function
        :param rec_func: recurrent connection function
        :param pseudo_grad_ops: pseudo gradient operation
        :param param: (current decay, voltage decay, voltage threshold, gradient parameter)
        """
        super(RecurrentCUBALIFCell, self).__init__()
        self.psp_func = psp_func
        self.rec_func = rec_func
        self.pseudo_grad_ops = pseudo_grad_ops
        self.cdecay, self.vdecay, self.vth, self.grad_win = param

    def forward(self, input_data, state):
        """
        :param input_data: input spike from pre-synaptic neurons
        :param state: (output spike of last timestep, current of last timestep, voltage of last timestep)
        :return: output spike, (output spike, current, voltage)
        """
        pre_spike, pre_current, pre_volt = state
        current = self.cdecay * pre_current + self.psp_func(input_data) + self.rec_func(pre_spike)
        volt = self.vdecay * pre_volt * (1. - pre_spike) + current
        output = self.pseudo_grad_ops(volt, self.vth, self.grad_win)
        return output, (output, current, volt)


class TemporalConvCUBALIFCell(nn.Module):
    def __init__(self, kernel, psp_func_list, pseudo_grad_ops, param):
        """
        :param kernel: kernel size of temporal conv
        :param psp_func_list: list of psp functions (same number as kernel)
        :param pseudo_grad_ops: pseudo gradient operation
        :param param: (current decay, voltage decay, voltage threshold, gradient parameter)
        """
        super(TemporalConvCUBALIFCell, self).__init__()
        self.kernel = kernel
        self.psp_func_list = nn.ModuleList(psp_func_list)
        self.pseudo_grad_ops = pseudo_grad_ops
        self.cdecay, self.vdecay, self.vth, self.grad_win = param

    def forward(self, input_data_list, state):
        """
        :param input_data_list: list of input spike from different timesteps (same number as kernel)
        :param state: (output spike of last timestep, current of last timestep, voltage of last timestep)
        :return: output spike, (output spike, current, voltage)
        """
        pre_spike, pre_current, pre_volt = state
        current = self.cdecay * pre_current
        if len(input_data_list) == self.kernel:
            for ts in range(self.kernel):
                current += self.psp_func_list[ts](input_data_list[ts])
        if len(input_data_list) < self.kernel:
            ts_diff = self.kernel - len(input_data_list)
            for ts in range(len(input_data_list)):
                current += self.psp_func_list[ts + ts_diff](input_data_list[ts])
        volt = self.vdecay * pre_volt * (1. - pre_spike) + current
        output = self.pseudo_grad_ops(volt, self.vth, self.grad_win)
        return output, (output, current, volt)


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
        pseudo_grad_ops_drop = PseudoSpikeRectDropout.apply

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
        self.fc1 = FeedForwardCUBALIFCellDropout(nn.Linear(c3, 128, bias=True),
                                          pseudo_grad_ops_drop,
                                          params)
        self.fc2 = nn.Linear(128, 2, bias=False)

    def forward(self, input_data, states):
        """
        :param input_data: input EEG spike trains
        :param states: list of (init spike, init voltage)
        :return: output
        """
        batch_size = input_data.shape[0]
        dropout_fc = DROPOUT_FC
        output_spikes = []
        temp_conv_spike_buffer = []
        c1_state, c2_state, c3_state, tc1_state, r1_state, f1_state = \
            states[0], states[1], states[2], states[3], states[4], states[5]

        mask_fc = Bernoulli(
            torch.full_like(torch.zeros(batch_size, 128, device=torch.device("cuda")), 1 - dropout_fc)).sample() / (
                       1 - dropout_fc)

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

            f1_spike, f1_state = self.fc1(r1_spike, f1_state, mask_fc, self.training)
            f2_output = self.fc2(f1_spike)
            output_spikes += [f2_output * self.ts_weights[step]]
        outputs = torch.stack(output_spikes).sum(dim=0)

        return outputs


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