import nxsdk.api.n2a as nx
import numpy as np


class PyTorch2LoihiNetwork:
    """
    A class of static helper functions to assist with implementing
    EEG SNN trained models on Loihi.
    """

    @classmethod
    def conv_layer(cls, pre_layer, bias_layer, neuron_spec, layer_spec, core_resource, fanin, fanout,
                   single_bias=True, bias_layer_num=1):
        """
        Create a new convolution layer on Loihi

        :param pre_layer: pre-synaptic layer of compartments
        :param bias_layer: layer of single bias neuron
        :param neuron_spec: dict spec for neurons in layer (vth, cdecay, vdecay)
        :param layer_spec: dict spec for layer (size_x, size_y, size_c, conv_x, conv_y, conv_c, stride_x, stride_y, raw_weight, raw_bias)
        :param core_resource: list of dict for available loihi core resource
        :param fanin: fanin connection number for single neuron of the layer
        :param fanout: fanout connection number for single neuron of the layer
        :param single_bias: if true only use single bias layer (if false then use > 1 bias layer to add precision)
        :param bias_layer_num: number of bias layer used
        :return: post_layer
        """
        # Propertied of the pre-synaptic layer
        net = pre_layer.net
        pre_size_x = pre_layer.sizeX
        pre_size_y = pre_layer.sizeY
        pre_size_c = pre_layer.sizeC

        # Optimize weight and bias for Loihi low precision
        conv_x = layer_spec['conv_x']
        conv_y = layer_spec['conv_y']
        conv_c = layer_spec['conv_c']
        stride_x = layer_spec['stride_x']
        stride_y = layer_spec['stride_y']
        w_b_dict, loihi_vth = cls.optimize_conv_layer_weight_bias(layer_spec['raw_weight'],
                                                                  layer_spec['raw_bias'],
                                                                  neuron_spec['vth'],
                                                                  conv_x, conv_y, conv_c)

        # Neuron prototype of the layer
        neuron_proto = nx.CompartmentPrototype(vThMant=loihi_vth,
                                               compartmentCurrentDecay=int((1 - neuron_spec['cdecay']) * 2 ** 12),
                                               compartmentVoltageDecay=int((1 - neuron_spec['vdecay']) * 2 ** 12))

        # Connection prototype of the layer
        conn_proto = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.MIXED,
                                            numWeightBits=8,
                                            numTagBits=0,
                                            numDelayBits=0,
                                            compressionMode=nx.SYNAPSE_COMPRESSION_MODE.DENSE)

        # Create compartment group of the layer
        post_size_x = layer_spec['size_x']
        post_size_y = layer_spec['size_y']
        post_size_c = layer_spec['size_c']
        post_dim = post_size_x * post_size_y * post_size_c
        post_layer_core_list = cls.compute_core_list(post_dim, fanin, fanout, core_resource)
        post_layer = net.createCompartmentGroup(size=post_dim, prototype=neuron_proto,
                                                logicalCoreId=post_layer_core_list)
        post_layer.sizeX = post_size_x
        post_layer.sizeY = post_size_y
        post_layer.sizeC = post_size_c
        post_layer.strideX = stride_x
        post_layer.strideY = stride_y

        # Setup Convolution Connection between Pre-synaptic layer and Post layer
        for kernel_x in range(0, pre_size_x - conv_x + 1, stride_x):
            for kernel_y in range(0, pre_size_y - conv_y + 1, stride_y):
                # Group Pre Block and Post Block for the kernel
                pre_kernel_block = net.createCompartmentGroup()
                for cc in range(pre_size_c):
                    for yy in range(kernel_y, kernel_y + conv_y):
                        for xx in range(kernel_x, kernel_x + conv_x):
                            pre_kernel_block.addCompartments(
                                pre_layer[cc * pre_size_y * pre_size_x + yy * pre_size_x + xx])
                post_kernel_block = net.createCompartmentGroup()
                post_kernel_x = kernel_x // stride_x
                post_kernel_y = kernel_y // stride_y
                for cc in range(post_size_c):
                    post_kernel_block.addCompartments(
                        post_layer[cc * post_size_y * post_size_x + post_kernel_y * post_size_x + post_kernel_x])
                # Setup Connections
                w_conn = pre_kernel_block.connect(post_kernel_block,
                                                  prototype=conn_proto,
                                                  weight=w_b_dict['weight'])
                if single_bias:
                    bias_conn = bias_layer.connect(post_kernel_block,
                                                   prototype=conn_proto,
                                                   weight=w_b_dict['bias'])
                else:
                    for nn in range(bias_layer_num):
                        bias_conn = bias_layer.connect(post_kernel_block,
                                                       prototype=conn_proto,
                                                       weight=w_b_dict['bias'])

        return post_layer

    @staticmethod
    def optimize_conv_layer_weight_bias(raw_weight, raw_bias, raw_vth, conv_x, conv_y, conv_c):
        """
        Transform weight and bias to Loihi weight and bias

        :param raw_weight: raw pytorch conv weight (post_channel, pre_channel, conv_y, conv_x)
        :param raw_bias: raw pytorch conv bias (post_channel,)
        :param raw_vth: raw voltage threshold
        :param conv_x: kernel size x
        :param conv_y: kernel size y
        :param conv_c: pre channel
        :return: weight_bias_dict, loihi_vth
        """
        raw_weight = raw_weight.reshape((-1, conv_x * conv_y * conv_c))
        raw_bias = raw_bias.reshape(-1, 1)
        max_w, min_w = np.amax(raw_weight), np.amin(raw_weight)
        max_b, min_b = np.amax(raw_bias), np.amin(raw_bias)
        max_abs_value = max(abs(max_w), abs(min_w), abs(max_b), abs(min_b))
        scale_factor = 255. / max_abs_value
        loihi_vth = int(raw_vth * scale_factor)
        loihi_weight = np.round(np.clip(raw_weight * scale_factor, -255, 255)).astype(int)
        loihi_bias = np.round(np.clip(raw_bias * scale_factor, -255, 255)).astype(int)
        weight_bias_dict = {'weight': loihi_weight, 'bias': loihi_bias}
        return weight_bias_dict, loihi_vth

    @staticmethod
    def combine_conv_layer_weight_w_avgpool(raw_weight, kernel_size=2):
        """
        Combining average pooling with conv layer and transform weight

        :param raw_weight: weight for conv layer after avg_pool (post_c, pre_c, conv_y, conv_x)
        :param kernel_size: average pooling kernel size
        :return: combine_weight
        """
        post_c, pre_c, conv_y, conv_x = raw_weight.shape[0], raw_weight.shape[1], raw_weight.shape[2], raw_weight.shape[3]
        combine_weight = np.zeros((post_c, pre_c, conv_y * kernel_size, conv_x * kernel_size))
        avg_pool_factor = 1. / kernel_size ** 2
        for y in range(conv_y):
            for x in range(conv_x):
                for avg_y in range(kernel_size):
                    for avg_x in range(kernel_size):
                        combine_weight[:, :, y * kernel_size + avg_y, x * kernel_size + avg_x] = avg_pool_factor * raw_weight[:,
                                                                                                              :, y, x]
        return combine_weight

    @classmethod
    def temp_conv_layer(cls, pre_layer, bias_layer, neuron_spec, layer_spec, core_resource, fanin, fanout):
        """
        Create a new temporal convolution layer

        :param pre_layer: pre-synaptic layer of compartments
        :param bias_layer: layer of single bias neuron
        :param neuron_spec: dict spec for neurons in layer (vth, cdecay, vdecay)
        :param layer_spec: dict spec for layer (dimension, window, raw_weight_list, raw_bias_list)
        :param core_resource: list of dict for available loihi core resource
        :param fanin: fanin connection number for single neuron of the layer
        :param fanout: fanout connection number for single neuron of the layer
        :return: post_layer
        """
        # Properties of the input layer
        net = pre_layer.net

        # Optimize weight and bias
        w_b_dict, loihi_vth = cls.optimize_temp_conv_layer_weight_bias(layer_spec['raw_weight_list'],
                                                                       layer_spec['raw_bias_list'],
                                                                       neuron_spec['vth'],
                                                                       layer_spec['window'])

        # Neuron prototype of the layer
        neuron_proto = nx.CompartmentPrototype(vThMant=loihi_vth,
                                               compartmentCurrentDecay=int((1 - neuron_spec['cdecay']) * 2 ** 12),
                                               compartmentVoltageDecay=int((1 - neuron_spec['vdecay']) * 2 ** 12))

        # Connection prototype of the layer
        conn_proto = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.MIXED,
                                            numWeightBits=8,
                                            numTagBits=0,
                                            compressionMode=nx.SYNAPSE_COMPRESSION_MODE.DENSE)

        # Create compartment group of the layer
        post_layer_core_list = cls.compute_core_list(layer_spec['dimension'], fanin, fanout, core_resource)
        post_layer = net.createCompartmentGroup(size=layer_spec['dimension'], prototype=neuron_proto,
                                                logicalCoreId=post_layer_core_list)

        # Connect layer with pre-synaptic layer, and bias layer
        for win in range(layer_spec['window']):
            w_conn = pre_layer.connect(post_layer,
                                       prototype=conn_proto,
                                       delay=w_b_dict['delay'][win],
                                       weight=w_b_dict['weight'][win])
            bias_conn = bias_layer[win].connect(post_layer,
                                                prototype=conn_proto,
                                                weight=w_b_dict['bias'][win])

        return post_layer

    @staticmethod
    def optimize_temp_conv_layer_weight_bias(raw_weight_list, raw_bias_list, raw_vth, window):
        """
        Transform weight and bias to loihi weight and bias (t-window-1, t-window-2, ..., t)

        :param raw_weight_list: list of raw pytorch weight
        :param raw_bias_list: list of raw pytorch bias
        :param raw_vth: raw voltage threshold
        :param window: temp conv window
        :return: weight_bias_dict, loihi_vth
        """
        max_abs_value_list = []
        for win in range(window):
            raw_bias_list[win] = raw_bias_list[win].reshape(-1, 1)
            max_w, min_w = np.amax(raw_weight_list[win]), np.amin(raw_weight_list[win])
            max_b, min_b = np.amax(raw_bias_list[win]), np.amin(raw_bias_list[win])
            max_abs_value = max(abs(max_w), abs(min_w), abs(max_b), abs(min_b))
            max_abs_value_list.append(max_abs_value)
        final_max_value = max(max_abs_value_list)
        scale_factor = 255. / final_max_value
        loihi_vth = int(raw_vth * scale_factor)
        weight_bias_dict = {'weight': [], 'delay': [], 'bias': []}
        for win in range(window):
            loihi_weight = np.clip(raw_weight_list[win] * scale_factor, -255, 255).astype(int)
            loihi_bias = np.clip(raw_bias_list[win] * scale_factor, -255, 255).astype(int)
            loihi_delay = np.zeros((loihi_weight.shape[0], loihi_weight.shape[1]), dtype=int)
            loihi_delay[loihi_weight != 0] = window - win - 1
            weight_bias_dict['delay'].append(loihi_delay)
            weight_bias_dict['weight'].append(loihi_weight)
            weight_bias_dict['bias'].append(loihi_bias)

        return weight_bias_dict, loihi_vth

    @classmethod
    def recurrent_layer_w_identity_input(cls, pre_layer, bias_layer, neuron_spec, layer_spec,
                                         core_resource, fanin, fanout):
        """
        Create a new recurrent layer (have identity connection with pre_layer and bias only for recurrent connection)

        :param pre_layer: pre-synaptic layer of compartments
        :param bias_layer: layer of single bias neuron
        :param neuron_spec: dict spec for neurons in layer (vth, cdecay, vdecay)
        :param layer_spec: dict spec for layer (dimension, raw_rec_weight, raw_rec_bias)
        :param core_resource: list of dict for available loihi core resource
        :param fanin: fanin connection number for single neuron of the layer
        :param fanout: fanout connection number for single neuron of the layer
        :return: post_layer
        """
        # Properties of the input layer
        net = pre_layer.net
        pre_dim = pre_layer.numNodes

        # Optimize weight and bias
        w_b_dict, loihi_vth = cls.optimize_recurrent_layer_weight_bias_with_identity(layer_spec['raw_rec_weight'],
                                                                                     layer_spec['raw_rec_bias'],
                                                                                     neuron_spec['vth'],
                                                                                     pre_dim, 4)
        # Neuron prototype of the layer
        neuron_proto = nx.CompartmentPrototype(vThMant=loihi_vth,
                                               compartmentCurrentDecay=int((1 - neuron_spec['cdecay']) * 2 ** 12),
                                               compartmentVoltageDecay=int((1 - neuron_spec['vdecay']) * 2 ** 12))

        # Connection prototype of the layer
        identity_conn_proto = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.EXCITATORY,
                                                     numWeightBits=8,
                                                     weightExponent=4,
                                                     numTagBits=0,
                                                     numDelayBits=0)
        conn_proto = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.MIXED,
                                            numWeightBits=8,
                                            numTagBits=0,
                                            numDelayBits=0,
                                            compressionMode=nx.SYNAPSE_COMPRESSION_MODE.DENSE)

        # Create compartment group of the layer
        post_layer_core_list = cls.compute_core_list(layer_spec['dimension'], fanin, fanout, core_resource)
        post_layer = net.createCompartmentGroup(size=layer_spec['dimension'], prototype=neuron_proto,
                                                logicalCoreId=post_layer_core_list)

        # Connect layer with pre-synaptic layer, recurrent connection, and bias layer
        identity_conn = pre_layer.connect(post_layer,
                                          prototype=identity_conn_proto,
                                          weight=w_b_dict['identity_w'],
                                          connectionMask=w_b_dict['identity_w_mask'])
        w_conn = post_layer.connect(post_layer,
                                    prototype=conn_proto,
                                    weight=w_b_dict['weight'])
        bias_conn = bias_layer.connect(post_layer,
                                       prototype=conn_proto,
                                       weight=w_b_dict['bias'])

        return post_layer

    @staticmethod
    def optimize_recurrent_layer_weight_bias_with_identity(raw_weight, raw_bias, raw_vth, pre_dim, identity_wexp):
        """
        Transform recurrent weight (with identity incoming) and bias to loihi weight and bias

        :param raw_weight: raw pytorch recurrent weight
        :param raw_bias: raw pytorch recurrent bias
        :param raw_vth: raw voltage threshold
        :param pre_dim: pre-synaptic layer dimension
        :param identity_wexp: weight exponent for identity weight
        :return: weight_bias_dict, loihi_vth
        """
        raw_bias = raw_bias.reshape(-1, 1)
        max_w, min_w = np.amax(raw_weight), np.amin(raw_weight)
        max_b, min_b = np.amax(raw_bias), np.amin(raw_bias)
        max_abs_value = max(abs(max_w), abs(min_w), abs(max_b), abs(min_b))
        scale_factor = 255. / max_abs_value
        loihi_vth = int(raw_vth * scale_factor)
        loihi_weight = np.clip(raw_weight * scale_factor, -255, 255).astype(int)
        loihi_bias = np.clip(raw_bias * scale_factor, -255, 255).astype(int)
        weight_bias_dict = {'weight': loihi_weight, 'bias': loihi_bias}

        # Compute identity weight and mask from pre-synaptic layer
        identity_w_mask = np.int_(np.eye(pre_dim))
        identity_w = np.clip(np.eye(pre_dim) * (scale_factor / 2 ** identity_wexp), 0, 255).astype(np.int)
        weight_bias_dict['identity_w'] = identity_w
        weight_bias_dict['identity_w_mask'] = identity_w_mask

        return weight_bias_dict, loihi_vth

    @classmethod
    def fully_connect_layer(cls, pre_layer, bias_layer, neuron_spec, layer_spec, core_resource, fanin, fanout):
        """
        Create a new fully connected layer

        :param pre_layer: pre-synaptic layer of compartments
        :param bias_layer: layer of single bias neuron
        :param neuron_spec: dict spec for neurons in layer (vth, cdecay, vdecay)
        :param layer_spec: dict spec for layer (dimension, raw_weight, raw_bias)
        :param core_resource: list of dict for available loihi core resource
        :param fanin: fanin connection number for single neuron of the layer
        :param fanout: fanout connection number for single neuron of the layer
        :return: post_layer
        """
        # Properties of the input layer
        net = pre_layer.net

        # Optimize weight and bias
        w_b_dict, loihi_vth = cls.optimize_full_layer_weight_bias(layer_spec['raw_weight'],
                                                                  layer_spec['raw_bias'],
                                                                  neuron_spec['vth'])

        # Neuron prototype of the layer
        neuron_proto = nx.CompartmentPrototype(vThMant=loihi_vth,
                                               compartmentCurrentDecay=int((1 - neuron_spec['cdecay']) * 2 ** 12),
                                               compartmentVoltageDecay=int((1 - neuron_spec['vdecay']) * 2 ** 12))

        # Connection prototype of the layer
        conn_proto = nx.ConnectionPrototype(signMode=nx.SYNAPSE_SIGN_MODE.MIXED,
                                            numWeightBits=8,
                                            numTagBits=0,
                                            numDelayBits=0,
                                            compressionMode=nx.SYNAPSE_COMPRESSION_MODE.DENSE)

        # Create compartment group of the layer
        post_layer_core_list = cls.compute_core_list(layer_spec['dimension'], fanin, fanout, core_resource)
        post_layer = net.createCompartmentGroup(size=layer_spec['dimension'], prototype=neuron_proto,
                                                logicalCoreId=post_layer_core_list)

        # Connect layer with pre-synaptic layer and bias layer
        w_conn = pre_layer.connect(post_layer,
                                   prototype=conn_proto,
                                   weight=w_b_dict['weight'])
        bias_conn = bias_layer.connect(post_layer,
                                       prototype=conn_proto,
                                       weight=w_b_dict['bias'])

        return post_layer

    @staticmethod
    def optimize_full_layer_weight_bias(raw_weight, raw_bias, raw_vth):
        """
        Transform weight and bias to loihi weight and bias

        :param raw_weight: raw pytorch weight
        :param raw_bias: raw pytorch bias
        :param raw_vth: raw voltage threshold
        :return: weight_bias_dict, loihi_vth
        """
        raw_bias = raw_bias.reshape(-1, 1)
        max_w, min_w = np.amax(raw_weight), np.amin(raw_weight)
        max_b, min_b = np.amax(raw_bias), np.amin(raw_bias)
        max_abs_value = max(abs(max_w), abs(min_w), abs(max_b), abs(min_b))
        scale_factor = 255. / max_abs_value
        loihi_vth = int(raw_vth * scale_factor)
        loihi_weight = np.clip(raw_weight * scale_factor, -255, 255).astype(int)
        loihi_bias = np.clip(raw_bias * scale_factor, -255, 255).astype(int)
        weight_bias_dict = {'weight': loihi_weight, 'bias': loihi_bias}
        return weight_bias_dict, loihi_vth

    @classmethod
    def bias_layers(cls, net, bias_layer_names, core_resource, fanout_list):
        """
        Create a list of bias neuron layers (each layer only have one bias neuron)

        :param net: NxNet
        :param bias_layer_names: list of name for each bias layer
        :param core_resource: list of dict for available loihi core resource
        :param fanout_list: list of fanout connection number for single neuron of the layer
        :return: bias_layer_dict
        """
        # Neuron prototype of bias layer
        neuron_proto = nx.CompartmentPrototype(vThMant=100,
                                               compartmentVoltageDecay=4095,
                                               compartmentCurrentDecay=4095)

        # Create bias layer for each layer of the network
        bias_layer_dict = {}
        for idx, name in enumerate(bias_layer_names, 0):
            bias_layer_core_list = cls.compute_core_list(1, 1, fanout_list[idx], core_resource)
            bias_layer = net.createCompartmentGroup(size=1, prototype=neuron_proto, logicalCoreId=bias_layer_core_list)
            bias_layer_dict[name] = bias_layer

        return bias_layer_dict

    @staticmethod
    def compute_core_list(neuron_dim, fanin, fanout, loihi_core_resource):
        """
        Compute core list base on loihi core resource
        :param neuron_dim: number of neurons
        :param fanin: incoming connections per neuron
        :param fanout: outgoing connections per neuron
        :param loihi_core_resource: list of dict for loihi core resource
        :return: core_list
        """
        core_num = len(loihi_core_resource)
        unassign_neuron = neuron_dim
        core_list = []
        for core in range(core_num):
            while unassign_neuron > 0 and loihi_core_resource[core]['compartment'] > 0 and loihi_core_resource[core][
                'fanin'] >= fanin and loihi_core_resource[core]['fanout'] >= fanout:
                core_list.append(core)
                unassign_neuron -= 1
                loihi_core_resource[core]['compartment'] -= 1
                loihi_core_resource[core]['fanin'] -= fanin
                loihi_core_resource[core]['fanout'] -= fanout
            if unassign_neuron == 0:
                break
        return core_list

    @staticmethod
    def analyze_core_usage(loihi_core_resource, compartment=14, fanin=16384, fanout=16384):
        """
        Analyze core usage

        :param loihi_core_resource: list of dict for loihi core resource
        :param compartment: max number of compartment
        :param fanin: max number of fanin
        :param fanout: max number of fanout
        :return: core_usage
        """
        core_usage = 0
        for core in range(len(loihi_core_resource)):
            if loihi_core_resource[core]['compartment'] < compartment:
                core_usage += 1
            elif loihi_core_resource[core]['fanin'] < fanin:
                core_usage += 1
            elif loihi_core_resource[core]['fanout'] < fanout:
                core_usage += 1

        return core_usage
