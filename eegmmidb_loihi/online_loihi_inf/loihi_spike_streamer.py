import nxsdk.api.n2a as nx
import numpy as np
import sys
sys.path.append('../')

from online_loihi_inf.loihi_network import PyTorch2LoihiNetwork
from spike_streamer.src.streaming import SpikeStreamer


class SpikeStreamer4LoihiNetwork:
    """
    Spike Streamer as Input to Loihi network for fast inference
    """

    @staticmethod
    def spike_streamer_1d_layer(net, dimension, core_resource, fanout,
                                spikes_per_packet=1024):
        """
        1-Dimensional input layer for SpikeGen (for testing 1D layers)

        :param net: NxNet
        :param dimension: dimension of the input
        :param core_resource: list of dict for available loihi core resource
        :param fanout: fanout connection number for single neuron of the layer
        :param spikes_per_packet: number of spikes per packet to sent to Loihi
        :return: input_layer, spike_streamer
        """
        input_layer_core_list = PyTorch2LoihiNetwork.compute_core_list(dimension, 1, fanout, core_resource)
        spike_streamer = SpikeStreamer(net)
        spike_streamer.setupSpikeInput(dimension, spikesPerPacket=spikes_per_packet,
                                       microsecondsPerTimestep=1, logicalCoreId=input_layer_core_list)
        return spike_streamer.inputLayer, spike_streamer

    @staticmethod
    def spike_streamer_2d_layer(net, shape, core_resource, fanout,
                                spikes_per_packet=1024):
        """
        2-Dimensional input layer for SpikeGen (for testing Conv layers)

        :param net: NxNet
        :param shape: shape of input CHW (c, y, x)
        :param core_resource: list of dict for available loihi core resource
        :param fanout: fanout connection number for single neuron of the layer
        :param spikes_per_packet: number of spikes per packet to sent to Loihi
        :return: input_layer, spike_streamer
        """
        size_x = shape['size_x']
        size_y = shape['size_y']
        size_c = shape['size_c']
        dimension = size_c * size_y * size_x
        input_layer_core_list = PyTorch2LoihiNetwork.compute_core_list(dimension, 1, fanout, core_resource)
        spike_streamer = SpikeStreamer(net)
        spike_streamer.setupSpikeInput(dimension, spikesPerPacket=spikes_per_packet,
                                       microsecondsPerTimestep=1, logicalCoreId=input_layer_core_list)
        spike_streamer.inputLayer.sizeX = size_x
        spike_streamer.inputLayer.sizeY = size_y
        spike_streamer.inputLayer.sizeC = size_c
        return spike_streamer.inputLayer, spike_streamer

    @staticmethod
    def add_input_spike_time(spike_streamer, spike_data, start_step):
        """
        Add spike data to spike streamer

        :param spike_streamer: spike streamer
        :param spike_data: spike data
        :param start_step: start step for the block of spike data
        """
        timestep = spike_data.shape[1]
        spike_neuron_list, spike_time_list = [], []
        for tt in range(timestep):
            tmp_spike_neuron = (np.where(spike_data[:, tt])[0]).tolist()
            tmp_spike_time = [tt + start_step for _ in range(len(tmp_spike_neuron))]
            spike_neuron_list.extend(tmp_spike_neuron)
            spike_time_list.extend(tmp_spike_time)
        spike_streamer.sendSpikes(spike_neuron_list, spike_time_list)
