import nxsdk.api.n2a as nx
from nxsdk.graph.monitor.probes import SpikeProbeCondition
import numpy as np
import os
import sys

sys.path.append('../')

from online_loihi_inf.loihi_network import PyTorch2LoihiNetwork


class OnlineInputOutput4LoihiNetwork:
    """ Online Input (bias) and Output (weighted F1) Setup for Loihi Network """

    @staticmethod
    def online_bias_layers(net, bias_layer_num, core_resource, fanout_list):
        """
        Create online bias layer

        :param net: NxNet
        :param bias_layer_num: number of bias neuron layers
        :param core_resource: list of dict for available loihi core resource
        :param fanout_list: list of fanout connection number for single neuron of the layer
        :return: bias_layer_list, pseudo_2_bias_list
        """
        # Neuron prototype of bias layer
        neuron_proto = nx.CompartmentPrototype(vThMant=100,
                                               compartmentVoltageDecay=4095,
                                               compartmentCurrentDecay=4095)
        pseudo_prototype = nx.CompartmentPrototype()

        # Create bias layer for each layer of the network
        bias_layer_list = []
        pseudo_2_bias_list = []
        for idx in range(bias_layer_num):
            bias_layer_core_list = PyTorch2LoihiNetwork.compute_core_list(1, 1, fanout_list[idx], core_resource)
            bias_layer = net.createCompartmentGroup(size=1, prototype=neuron_proto, logicalCoreId=bias_layer_core_list)
            pseudo_layer = net.createCompartmentGroup(size=1, prototype=pseudo_prototype,
                                                      logicalCoreId=bias_layer_core_list)
            conn_w = np.eye(1) * 120
            conn_mask = np.int_(np.eye(1))
            pseudo_2_bias = pseudo_layer.connect(bias_layer,
                                                 prototype=nx.ConnectionPrototype(
                                                     signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY),
                                                 connectionMask=conn_mask,
                                                 weight=conn_w)
            bias_layer_list.append(bias_layer)
            pseudo_2_bias_list.append(pseudo_2_bias)

        return bias_layer_list, pseudo_2_bias_list

    @staticmethod
    def get_online_bias_layers_axon_id(net, pseudo_2_bias_list):
        """
        Get axon id for online bias layers

        :param net: NxNet
        :param pseudo_2_bias_list: pseudo connection list to online bias layers
        :return: bias_layer_core_id_list, bias_layer_axon_id_list
        """
        bias_layer_core_id_list = []
        bias_layer_axon_id_list = []
        for pseudo_2_bias in pseudo_2_bias_list:
            for conn in pseudo_2_bias:
                raw_axon_id = net.resourceMap.inputAxon(conn.inputAxon.nodeId)
                bias_layer_core_id_list.append(raw_axon_id[0][2])
                bias_layer_axon_id_list.append(raw_axon_id[0][3])

        return bias_layer_core_id_list, bias_layer_axon_id_list

    @classmethod
    def generate_bias_encoding_header(cls, snip_dir, net, pseudo_2_bias_list, bias_spike_info_list,
                                      spike_ts, extra_ts):
        """
        Generate the additional information header file for bias input encoding

        :param snip_dir: SNIP file directory
        :param net: NxNet
        :param pseudo_2_bias_list: pseudo connection list to online bias layers
        :param bias_spike_info_list: list of bias spike information (start step, end step)
        :param spike_ts: spike timesteps
        :param extra_ts: extra timesteps shift with layer operations

        """
        bias_layer_core_id_list, bias_layer_axon_id_list = cls.get_online_bias_layers_axon_id(net, pseudo_2_bias_list)
        bias_dim = len(bias_spike_info_list)

        # Create header file
        abs_snip_dir = os.path.abspath(snip_dir)
        bias_info_header_path = abs_snip_dir + '/bias_encoding_info.h'
        f = open(bias_info_header_path, 'w')
        f.write('/* Temporary generated file for defining parameters for bias encoding*/\n')

        # Write basic information
        f.write('#define BIAS_DIMENSION ' + str(bias_dim) + '\n')
        f.write('#define WINDOW_STEP ' + str(spike_ts + extra_ts) + '\n')
        f.write('#define BIAS_CHIP_ID 0\n')

        # Write axon information
        f.write('static int bias_core_id_list[' + str(bias_dim) + ']={' +
                ','.join([str(ii) for ii in bias_layer_core_id_list]) + '};\n')
        f.write('static int bias_axon_id_list[' + str(bias_dim) + ']={' +
                ','.join([str(ii) for ii in bias_layer_axon_id_list]) + '};\n')

        # Write bias step spike information
        f.write('static int bias_start_list[' + str(bias_dim) + ']={' +
                ','.join([str(ii[0]) for ii in bias_spike_info_list]) + '};\n')
        f.write('static int bias_end_list[' + str(bias_dim) + ']={' +
                ','.join([str(ii[1]) for ii in bias_spike_info_list]) + '};\n')

        f.close()

    @classmethod
    def setup_bias_encoding_snip(cls, snip_dir, board, net, pseudo_2_bias_list, bias_spike_info_list,
                                 spike_ts, extra_ts):
        """
        Setup bias encoding SNIP

        :param snip_dir: SNIP file directory
        :param board: compiled Loihi network
        :param net: NxNet
        :param pseudo_2_bias_list: pseudo connection list to online bias layers
        :param bias_spike_info_list: list of bias spike information (start step, end step)
        :param spike_ts: spike timesteps
        :param extra_ts: extra timesteps shift with layer operations
        """
        abs_snip_dir = os.path.abspath(snip_dir)
        cls.generate_bias_encoding_header(snip_dir, net, pseudo_2_bias_list, bias_spike_info_list,
                                          spike_ts, extra_ts)
        bias_encoder_snip = board.createProcess(name='bias_encoding',
                                                includeDir=abs_snip_dir,
                                                cFilePath=abs_snip_dir + '/bias_encoding.c',
                                                funcName='run_encoder',
                                                guardName='do_encoder',
                                                phase="spiking",
                                                chipId=0,
                                                lmtId=1)

    @staticmethod
    def online_fc_layer(net, neuron_spec, layer_spec, core_resource, fanin, fanout):
        """
        Define online FC layer without pre-synaptic layer connections

        :param net: NxNet
        :param neuron_spec: dict spec for neurons in layer (vth, cdecay, vdecay)
        :param layer_spec: dict spec for layer (dimension, raw_weight, raw_bias)
        :param core_resource: list of dict for available loihi core resource
        :param fanin: fanin connection number for single neuron of the layer
        :param fanout: fanout connection number for single neuron of the layer
        :return: post_layer, w_b_dict
        """
        # Optimize weight and bias
        w_b_dict, loihi_vth = PyTorch2LoihiNetwork.optimize_full_layer_weight_bias(layer_spec['raw_weight'],
                                                                                   layer_spec['raw_bias'],
                                                                                   neuron_spec['vth'])

        # Neuron prototype of the layer
        neuron_proto = nx.CompartmentPrototype(vThMant=loihi_vth,
                                               compartmentCurrentDecay=int((1 - neuron_spec['cdecay']) * 2 ** 12),
                                               compartmentVoltageDecay=int((1 - neuron_spec['vdecay']) * 2 ** 12))

        # Create compartment group of the layer
        post_layer_core_list = PyTorch2LoihiNetwork.compute_core_list(layer_spec['dimension'], fanin, fanout,
                                                                      core_resource)
        post_layer = net.createCompartmentGroup(size=layer_spec['dimension'], prototype=neuron_proto,
                                                logicalCoreId=post_layer_core_list)

        return post_layer, w_b_dict

    @staticmethod
    def online_fc_layer_connection(pre_layer, bias_layer, post_layer, weight_bias_dict):
        """
        Connect pre-synaptic layer and bias layer to FC layer

        :param pre_layer: pre-synaptic layer of compartments
        :param bias_layer: layer of single bias neuron
        :param post_layer: post-synaptic layer of compartments
        :param weight_bias_dict: weight and bias

        """
        # Connection prototype of the layer
        conn_proto = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.MIXED,
                                            numWeightBits=8,
                                            numTagBits=0,
                                            numDelayBits=0,
                                            compressionMode=nx.SYNAPSE_COMPRESSION_MODE.DENSE)

        # Connect layer with pre-synaptic layer and bias layer
        w_conn = pre_layer.connect(post_layer,
                                   prototype=conn_proto,
                                   weight=weight_bias_dict['weight'])
        bias_conn = bias_layer.connect(post_layer,
                                       prototype=conn_proto,
                                       weight=weight_bias_dict['bias'])

    @staticmethod
    def online_fc_decoding(fc_layer):
        """
        Create Pseudo spike probe for online FC layer decoding

        :param fc_layer: fully-connected layer
        """
        custom_probe_cond = SpikeProbeCondition(tStart=10000000000)
        pseudo_spike_probe = fc_layer.probe(nx.ProbeParameter.SPIKE, custom_probe_cond)

    @staticmethod
    def generate_fc_decoding_header(snip_dir, fc_dim, raw_ts_weight, decode_start, decode_end, spike_ts, extra_ts,
                                    ts_weight_factor=10000):
        """
        Generate the additional information header file for FC layer decoding

        :param snip_dir: SNIP file directory
        :param fc_dim: FC layer dimension
        :param raw_ts_weight: raw timestep weights
        :param decode_start: decode start step
        :param decode_end: decode end step
        :param spike_ts: spike timesteps
        :param extra_ts: extra timesteps shift with layer operations
        :param ts_weight_factor: factor to make ts weight from float to int

        """
        # Convert ts weight from float to int
        ts_weight = (raw_ts_weight.squeeze() * ts_weight_factor).astype(int).tolist()

        # Create header file
        abs_snip_dir = os.path.abspath(snip_dir)
        fc_info_header_path = abs_snip_dir + '/fc_decoding_info.h'
        f = open(fc_info_header_path, 'w')
        f.write('/* Temporary generated file for defining parameters for fc decoding*/\n')

        # Write basic information
        f.write('#define FC_DIMENSION ' + str(fc_dim) + '\n')
        f.write('#define WINDOW_STEP ' + str(spike_ts + extra_ts) + '\n')
        f.write('#define DECODE_START_STEP ' + str(decode_start) + '\n')
        f.write('#define DECODE_END_STEP ' + str(decode_end) + '\n')

        # Write ts weight
        f.write('static int ts_weight[' + str(spike_ts) + ']={' + ','.join([str(ii) for ii in ts_weight]) + '};\n')

        f.close()

    @classmethod
    def setup_fc_decoding_snip(cls, snip_dir, board, fc_dim, raw_ts_weight, decode_start, decode_end,
                               spike_ts, extra_ts):
        """
        Setup FC decoding SNIP

        :param snip_dir: SNIP file directory
        :param board: compiled Loihi network
        :param fc_dim: FC layer dimension
        :param raw_ts_weight: raw timestep weights
        :param decode_start: decode start step
        :param decode_end: decode end step
        :param spike_ts: spike timesteps
        :param extra_ts: extra timesteps shift with layer operations
        :return: fc_decoder_channel
        """
        abs_snip_dir = os.path.abspath(snip_dir)
        cls.generate_fc_decoding_header(snip_dir, fc_dim, raw_ts_weight, decode_start, decode_end, spike_ts, extra_ts)
        fc_decoder_snip = board.createProcess(name='fc_decoding',
                                              includeDir=abs_snip_dir,
                                              cFilePath=abs_snip_dir + '/fc_decoding.c',
                                              funcName='run_decoder',
                                              guardName='do_decoder',
                                              phase="mgmt",
                                              chipId=0,
                                              lmtId=0)
        fc_decoder_channel = board.createChannel(b'decodeoutput', "int", fc_dim)
        fc_decoder_channel.connect(fc_decoder_snip, None)

        return fc_decoder_channel
