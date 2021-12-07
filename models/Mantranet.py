import torch
from torch import nn
from torch.nn import functional as F
from collections import OrderedDict

import numpy as np

# class ConvLSTMCell(nn.Module):
#
#     def __init__(self, input_dim, hidden_dim, kernel_size, bias):
#         """
#         Initialize ConvLSTM cell.
#         Parameters
#         ----------
#         input_dim: int
#             Number of channels of input tensor.
#         hidden_dim: int
#             Number of channels of hidden state.
#         kernel_size: (int, int)
#             Size of the convolutional kernel.
#         bias: bool
#             Whether or not to add the bias.
#         """
#
#         super(ConvLSTMCell, self).__init__()
#
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#
#         self.kernel_size = kernel_size
#         self.padding = kernel_size[0] // 2, kernel_size[1] // 2
#         self.bias = bias
#
#         self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
#                               out_channels=4 * self.hidden_dim,
#                               kernel_size=self.kernel_size,
#                               padding=self.padding,
#                               bias=self.bias)
#
#     def forward(self, input_tensor, cur_state):
#         h_cur, c_cur = cur_state
#
#         combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
#
#         combined_conv = self.conv(combined)
#         cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
#         i = torch.sigmoid(cc_i)
#         f = torch.sigmoid(cc_f)
#         o = torch.sigmoid(cc_o)
#         g = torch.tanh(cc_g)
#
#         c_next = f * c_cur + i * g
#         h_next = o * torch.tanh(c_next)
#
#         return h_next, c_next
#
#     def init_hidden(self, batch_size, image_size):
#         height, width = image_size
#         return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
#                 torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))
#
#
# class ConvLSTM(nn.Module):
#
#     """
#     Parameters:
#         input_dim: Number of channels in input
#         hidden_dim: Number of hidden channels
#         kernel_size: Size of kernel in convolutions
#         num_layers: Number of LSTM layers stacked on each other
#         batch_first: Whether or not dimension 0 is the batch or not
#         bias: Bias or no bias in Convolution
#         return_all_layers: Return the list of computations for all layers
#         Note: Will do same padding.
#     Input:
#         A tensor of size B, T, C, H, W or T, B, C, H, W
#     Output:
#         A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
#             0 - layer_output_list is the list of lists of length T of each output
#             1 - last_state_list is the list of last states
#                     each element of the list is a tuple (h, c) for hidden state and memory
#     Example:
#         >> x = torch.rand((32, 10, 64, 128, 128))
#         >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
#         >> _, last_states = convlstm(x)
#         >> h = last_states[0][0]  # 0 for layer index, 0 for h index
#     """
#
#     def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
#                  batch_first=False, bias=True, return_all_layers=False):
#         super(ConvLSTM, self).__init__()
#
#         self._check_kernel_size_consistency(kernel_size)
#
#         # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
#         kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
#         hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
#         if not len(kernel_size) == len(hidden_dim) == num_layers:
#             raise ValueError('Inconsistent list length.')
#
#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.kernel_size = kernel_size
#         self.num_layers = num_layers
#         self.batch_first = batch_first
#         self.bias = bias
#         self.return_all_layers = return_all_layers
#
#         cell_list = []
#         for i in range(0, self.num_layers):
#             cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
#
#             cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
#                                           hidden_dim=self.hidden_dim[i],
#                                           kernel_size=self.kernel_size[i],
#                                           bias=self.bias))
#
#         self.cell_list = nn.ModuleList(cell_list)
#
#     def forward(self, input_tensor, hidden_state=None):
#         """
#         Parameters
#         ----------
#         input_tensor: todo
#             5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
#         hidden_state: todo
#             None. todo implement stateful
#         Returns
#         -------
#         last_state_list, layer_output
#         """
#         if not self.batch_first:
#             # (t, b, c, h, w) -> (b, t, c, h, w)
#             input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
#
#         b, _, _, h, w = input_tensor.size()
#
#         # Implement stateful ConvLSTM
#         if hidden_state is not None:
#             raise NotImplementedError()
#         else:
#             # Since the init is done in forward. Can send image size here
#             hidden_state = self._init_hidden(batch_size=b,
#                                              image_size=(h, w))
#
#         layer_output_list = []
#         last_state_list = []
#
#         seq_len = input_tensor.size(1)
#         cur_layer_input = input_tensor
#
#         for layer_idx in range(self.num_layers):
#
#             h, c = hidden_state[layer_idx]
#             output_inner = []
#             for t in range(seq_len):
#                 h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
#                                                  cur_state=[h, c])
#                 output_inner.append(h)
#
#             layer_output = torch.stack(output_inner, dim=1)
#             cur_layer_input = layer_output
#
#             layer_output_list.append(layer_output)
#             last_state_list.append([h, c])
#
#         if not self.return_all_layers:
#             layer_output_list = layer_output_list[-1:]
#             last_state_list = last_state_list[-1:]
#
#         return layer_output_list, last_state_list
#
#     def _init_hidden(self, batch_size, image_size):
#         init_states = []
#         for i in range(self.num_layers):
#             init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
#         return init_states
#
#     @staticmethod
#     def _check_kernel_size_consistency(kernel_size):
#         if not (isinstance(kernel_size, tuple) or
#                 (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
#             raise ValueError('`kernel_size` must be tuple or list of tuples')
#
#     @staticmethod
#     def _extend_for_multilayer(param, num_layers):
#         if not isinstance(param, list):
#             param = [param] * num_layers
#         return param
#
# class SRMConv2D(nn.Module):
#     def __init__(self, in_channels, out_channels, stride=1, padding=2):
#         super(SRMConv2D, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.stride = stride
#         self.padding = padding
#         self.SRMWeights = nn.Parameter(
#             self._get_srm_list(), requires_grad=False)
#
#     def _get_srm_list(self):
#         # srm kernel 1
#         srm1 = [[0,  0, 0,  0, 0],
#                 [0, -1, 2, -1, 0],
#                 [0,  2, -4, 2, 0],
#                 [0, -1, 2, -1, 0],
#                 [0,  0, 0,  0, 0]]
#         srm1 = torch.tensor(srm1, dtype=torch.float32) / 4.
#
#         # srm kernel 2
#         srm2 = [[-1, 2, -2, 2, -1],
#                 [2, -6, 8, -6, 2],
#                 [-2, 8, -12, 8, -2],
#                 [2, -6, 8, -6, 2],
#                 [-1, 2, -2, 2, -1]]
#         srm2 = torch.tensor(srm2, dtype=torch.float32) / 12.
#
#         # srm kernel 3
#         srm3 = [[0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0],
#                 [0, 1, -2, 1, 0],
#                 [0, 0, 0, 0, 0],
#                 [0, 0, 0, 0, 0]]
#         srm3 = torch.tensor(srm3, dtype=torch.float32) / 2.
#
#         return torch.stack([torch.stack([srm1, srm1, srm1], dim=0), torch.stack([srm2, srm2, srm2], dim=0), torch.stack([srm3, srm3, srm3], dim=0)], dim=0)
#
#     def forward(self, X):
#         # X1 =
#         return F.conv2d(X, self.SRMWeights, stride=self.stride, padding=self.padding)
#
#
# class CombinedConv2D(nn.Module):
#     def __init__(self, in_channels=3):
#         super(CombinedConv2D, self).__init__()
#         self.conv2d = nn.Conv2d(
#             in_channels=in_channels, out_channels=10, stride=1, kernel_size=5, padding=2)
#         self.bayarConv2d = nn.Conv2d(
#             in_channels=in_channels, out_channels=3, stride=1, kernel_size=5, padding=2)
#         self.SRMConv2d = SRMConv2D(
#             in_channels=3, out_channels=3, stride=1, padding=2)
#
#     def forward(self, X):
#         X1 = F.relu(self.conv2d(X))
#         X2 = F.relu(self.bayarConv2d(X))
#         X3 = F.relu(self.SRMConv2d(X))
#         return torch.cat([X1, X2, X3], dim=1)
#
#
# class FeatexVGG16(nn.Module):
#     def __init__(self, type=1):
#         super(FeatexVGG16, self).__init__()
#         # block1
#         self.combinedConv = CombinedConv2D(in_channels=3)
#         self.block1 = nn.Sequential(OrderedDict([
#             ('b1c1', nn.Conv2d(
#                 in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)),
#             ('b1ac', nn.ReLU())
#         ]))
#
#         # block2
#         self.block2 = nn.Sequential(OrderedDict([
#             ('b2c1', nn.Conv2d(
#                 in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)),
#             ('b2ac1', nn.ReLU()),
#             ('b2c2', nn.Conv2d(in_channels=64,
#                                out_channels=64, kernel_size=3, stride=1, padding=1)),
#             ('b2ac2', nn.ReLU())
#         ]))
#
#         # block3
#         self.block3 = nn.Sequential(OrderedDict([
#             ('b3c1', nn.Conv2d(
#                 in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)),
#             ('b3ac1', nn.ReLU()),
#             ('b3c2', nn.Conv2d(in_channels=128, kernel_size=3,
#                                out_channels=128, stride=1, padding=1)),
#             ('b3ac2', nn.ReLU()),
#             ('b3c3', nn.Conv2d(in_channels=128,
#                                out_channels=128, kernel_size=3, stride=1, padding=1)),
#             ('b3ac3', nn.ReLU())
#         ]))
#
#         # block4
#         self.block4 = nn.Sequential(OrderedDict([
#             ('b4c1', nn.Conv2d(
#                 in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)),
#             ('b4ac1', nn.ReLU()),
#             ('b4c2', nn.Conv2d(in_channels=256,
#                                out_channels=256, kernel_size=3, stride=1, padding=1)),
#             ('b4ac2', nn.ReLU()),
#             ('b4c3', nn.Conv2d(in_channels=256,
#                                out_channels=256, kernel_size=3, stride=1, padding=1)),
#             ('b4ac3', nn.ReLU())
#         ]))
#
#         # block5
#         self.block5 = nn.Sequential(OrderedDict([
#             ('b5c1', nn.Conv2d(
#                 in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)),
#             ('b5ac1', nn.ReLU()),
#             ('b5c2', nn.Conv2d(in_channels=256,
#                                out_channels=256, kernel_size=3, stride=1, padding=1)),
#             ('b5ac2', nn.ReLU())
#         ]))
#
#         self.transform = nn.Conv2d(
#             in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
#         self.activation = None if type >= 1 else nn.Tanh()
#
#     def forward(self, X):
#         X= self.combinedConv(X)
#         X = self.block1(X)
#         X = self.block2(X)
#         X = self.block3(X)
#         X = self.block4(X)
#         X = self.block5(X)
#         X = self.transform(X)
#         if self.activation is not None:
#             X = self.activation(X)
#         return nn.functional.normalize(X, 2, dim=-1)
#
#
# class ZPool2D(nn.Module):
#     def __init__(self, kernel_size):
#         super(ZPool2D, self).__init__()
#         self.avgpool = nn.AvgPool2d(
#             kernel_size=kernel_size, stride=1, padding=kernel_size//2)
#
#     def forward(self, X):
#         mu = self.avgpool(X)
#         sigma = torch.sqrt((torch.pow(X, 2) - torch.pow(mu, 2)
#                             ).sum() / (X.shape[-2] * X.shape[-1]))
#         D = X - mu
#         return D / sigma
#
#
# class ZPool2DGlobal(nn.Module):
#     def __init__(self, size=[1, 64, 1, 1], epsilon=1e-5):
#         super(ZPool2DGlobal, self).__init__()
#         self.epsilon = epsilon
#         self.weight = nn.Parameter(torch.zeros(size), requires_grad=True)
#
#     def forward(self, X):
#         mu = torch.mean(X, dim=(2, 3), keepdim=True)
#         D = X - mu
#         sigma = torch.sqrt((torch.pow(X, 2) - torch.pow(mu, 2)
#                             ).sum(dim=(-1, -2), keepdim=True) / (X.shape[-2] * X.shape[-1]))
#         sigma = torch.max(sigma, self.epsilon + self.weight)
#         return D / sigma
#
#
# class MantraNet(nn.Module):
#     def __init__(self, Featex=FeatexVGG16(), pool_size_list=[7, 15, 31]):
#         super(MantraNet, self).__init__()
#         self.rf = Featex
#         self.outlierTrans = nn.Conv2d(
#             in_channels=256, out_channels=64, kernel_size=(1, 1), bias=False)
#         self.bnorm = nn.BatchNorm2d(num_features=64)
#         self.zpoolglobal = ZPool2DGlobal()
#         zpools = OrderedDict()
#         for i in pool_size_list:
#             name = 'ZPool2D@{}x{}'.format(i, i)
#             zpools[name] = ZPool2D(i)
#         self.zpools = nn.Sequential(zpools)
#         self.cLSTM = ConvLSTM(64, 8, (7, 7), 1, batch_first=True)
#         self.pred = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=7, padding=3)
#
#     def forward(self, X): # [B, 3, H, W]
#         # if self.rf is not None:
#         X = self.rf(X) # [B, 256, H, W]
#         X = self.bnorm(self.outlierTrans(X))
#         Z = []
#         Z.append(torch.unsqueeze(self.zpoolglobal(X), dim=1))
#         for index in range(len(self.zpools)-1, -1, -1):
#             Z.append(torch.unsqueeze(self.zpools[index](X), dim=1))
#         Z = torch.cat([i for i in Z], dim=1)
#         last_output_list, _ = self.cLSTM(Z)
#         X = last_output_list[0][:, -1, :, :, :]
#         # output = self.sigmoid(self.pred(X))
#         output = self.pred(X)
#         return output
#
#
# class IMTFE(nn.Module):
#     def __init__(self, Featex, in_size) -> None:
#         super(IMTFE, self).__init__()
#         self.Featex = Featex
#         self.conv1 = nn.Conv2d(
#             in_channels=256, out_channels=8, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(in_channels=8, out_channels=8,
#                                kernel_size=in_size, stride=1, padding=0)
#
#     def forward(self, input):
#         out = self.Featex(input)
#         out = self.conv1(out)
#         out = self.conv2(out)
#         return out
#
#     def getFeatex(self):
#         return self.Featex
#
#
# def bayarConstraint(weight_):
#     weight = weight_[0]
#     h, w = weight.shape[1: 3]
#     weight[:, h//2, w//2] = 0
#     weight /= weight.sum(dim=(1, 2), keepdim=True)
#     weight[:, h//2, w//2] = -1
#     return weight_
#
#
# X = torch.randn([2, 3, 256, 256])
# net = FeatexVGG16()
# # net = IMTFE(Featex=FeatexVGG16(), in_size=128)
# net = MantraNet(FeatexVGG16())
# # Y = net(X)
# # print(Y.shape)
# # net = CombinedConv2D()
# # weight = net.bayarConv2d.weight[0]
# # net.bayarConv2d.weight[0] = bayarConstraint(weight)
# # print(net.bayarConv2d.weight[0,:, 3,3])
# model = MantraNet(FeatexVGG16())
# out = model(X)
# print(out.shape)

# 다른 놈 코드

def hardsigmoid(T):
    T_0 = T
    T = 0.2 * T_0 + 0.5
    T[T_0 < -2.5] = 0
    T[T_0 > 2.5] = 1

    return T

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        self.sigmoid = hardsigmoid

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_c, cc_o = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = self.sigmoid(cc_i)
        f = self.sigmoid(cc_f)
        c_next = f * c_cur + i * torch.tanh(cc_c)
        o = self.sigmoid(cc_o)

        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))

class ConvLSTM(nn.Module):
    """
    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.transpose(0, 1)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

def reflect(x, minx, maxx):
    """ Reflects an array around two points making a triangular waveform that ramps up
    and down,  allowing for pad lengths greater than the input length """
    rng = maxx - minx
    double_rng = 2 * rng
    mod = np.fmod(x - minx, double_rng)
    normed_mod = np.where(mod < 0, mod + double_rng, mod)
    out = np.where(normed_mod >= rng, double_rng - normed_mod, normed_mod) + minx
    return np.array(out, dtype=x.dtype)

def symm_pad(im, padding):
    h, w = im.shape[-2:]
    left, right, top, bottom = padding

    x_idx = np.arange(-left, w + right)
    y_idx = np.arange(-top, h + bottom)

    x_pad = reflect(x_idx, -0.5, w - 0.5)
    y_pad = reflect(y_idx, -0.5, h - 0.5)
    xx, yy = np.meshgrid(x_pad, y_pad)
    return im[..., yy, xx]

def batch_norm(X, eps=0.001):
    # extract the dimensions
    N, C, H, W = X.shape
    device=X.device
    # mini-batch mean
    mean = X.mean(axis=(0, 2, 3)).to(device)
    # mini-batch variance
    variance = ((X - mean.view((1, C, 1, 1))) ** 2).mean(axis=(0, 2, 3)).to(device)
    # normalize
    X = (X - mean.reshape((1, C, 1, 1))) * 1.0 / torch.pow((variance.view((1, C, 1, 1)) + eps), 0.5)
    return X.to(device)

# MantraNet (equivalent from the one coded in tensorflow at https://github.com/ISICV/ManTraNet)
class MantraNet(nn.Module):
    def __init__(self, device, in_channel=3, eps=10 ** (-6)):
        super(MantraNet, self).__init__()

        self.eps = eps
        self.relu = nn.ReLU()
        self.device = device

        # ********** IMAGE MANIPULATION TRACE FEATURE EXTRACTOR *********

        ## Initialisation

        self.init_conv = nn.Conv2d(in_channel, 4, 5, 1, padding=0, bias=False)

        self.BayarConv2D = nn.Conv2d(in_channel, 3, 5, 1, padding=0, bias=False)
        self.bayar_mask = (torch.tensor(np.ones(shape=(5, 5))))#.to(self.device)
        self.bayar_mask[2, 2] = 0

        self.bayar_final = (torch.tensor(np.zeros((5, 5))))#.to(self.device)
        self.bayar_final[2, 2] = -1

        self.SRMConv2D = nn.Conv2d(in_channel, 9, 5, 1, padding=0, bias=False)
        self.SRMConv2D.weight.data = torch.load('MantraNetv4.pt')['SRMConv2D.weight']

        ##SRM filters (fixed)
        for param in self.SRMConv2D.parameters():
            param.requires_grad = False

        # print(torch.load('MantraNetv4.pt'))
        self.middle_and_last_block = nn.ModuleList([
            nn.Conv2d(16, 32, 3, 1, padding=0),#.load_state_dict(torch.load('MantraNetv4.pt')[4]),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, padding=0)]
        )

        # ********** LOCAL ANOMALY DETECTOR *********

        self.adaptation = nn.Conv2d(256, 64, 1, 1, padding=0, bias=False)

        self.sigma_F = nn.Parameter(torch.zeros((1, 64, 1, 1)), requires_grad=True)

        self.pool31 = nn.AvgPool2d(31, stride=1, padding=15, count_include_pad=False)
        self.pool15 = nn.AvgPool2d(15, stride=1, padding=7, count_include_pad=False)
        self.pool7 = nn.AvgPool2d(7, stride=1, padding=3, count_include_pad=False)

        self.convlstm = ConvLSTM(input_dim=64,
                                 hidden_dim=8,
                                 kernel_size=(7, 7),
                                 num_layers=1,
                                 batch_first=False,
                                 bias=True,
                                 return_all_layers=False)

        self.end = nn.Conv2d(8, 1, 7, 1, padding=3)
        # self.end = nn.Sequential(nn.Conv2d(8, 1, 7, 1, padding=3))

    def forward(self, x):
        B, nb_channel, H, W = x.shape

        if not (self.training):
            self.GlobalPool = nn.AvgPool2d((H, W), stride=1)
        else:
            if not hasattr(self, 'GlobalPool'):
                self.GlobalPool = nn.AvgPool2d((H, W), stride=1)

        # Normalization
        x = x / 255. * 2 - 1

        ## Image Manipulation Trace Feature Extractor

        ## **Bayar constraints**

        self.bayar_mask = self.bayar_mask.to(self.BayarConv2D.weight.data.device)
        self.BayarConv2D.weight.data *= self.bayar_mask
        self.BayarConv2D.weight.data *= torch.pow(self.BayarConv2D.weight.data.sum(axis=(2, 3)).view(3, 3, 1, 1), -1)
        self.bayar_final = self.bayar_final.to(self.BayarConv2D.weight.data.device)
        self.BayarConv2D.weight.data += self.bayar_final

        # Symmetric padding
        x = symm_pad(x, (2, 2, 2, 2))

        conv_init = self.init_conv(x)
        conv_bayar = self.BayarConv2D(x)
        conv_srm = self.SRMConv2D(x)

        first_block = torch.cat([conv_init, conv_srm, conv_bayar], axis=1)
        first_block = self.relu(first_block)

        last_block = first_block

        for layer in self.middle_and_last_block:

            if isinstance(layer, nn.Conv2d):
                last_block = symm_pad(last_block, (1, 1, 1, 1))

            last_block = layer(last_block)

        # L2 normalization
        last_block = F.normalize(last_block, dim=1, p=2)

        ## Local Anomaly Feature Extraction
        X_adapt = self.adaptation(last_block)
        X_adapt = batch_norm(X_adapt)

        # Z-pool concatenation
        mu_T = self.GlobalPool(X_adapt)
        sigma_T = torch.sqrt(self.GlobalPool(torch.square(X_adapt - mu_T)))
        sigma_T = torch.max(sigma_T, self.sigma_F + self.eps)
        inv_sigma_T = torch.pow(sigma_T, -1)
        zpoolglobal = torch.abs((mu_T - X_adapt) * inv_sigma_T)

        mu_31 = self.pool31(X_adapt)
        zpool31 = torch.abs((mu_31 - X_adapt) * inv_sigma_T)

        mu_15 = self.pool15(X_adapt)
        zpool15 = torch.abs((mu_15 - X_adapt) * inv_sigma_T)

        mu_7 = self.pool7(X_adapt)
        zpool7 = torch.abs((mu_7 - X_adapt) * inv_sigma_T)

        input_lstm = torch.cat(
            [zpool7.unsqueeze(0), zpool15.unsqueeze(0), zpool31.unsqueeze(0), zpoolglobal.unsqueeze(0)], axis=0)

        # Conv2DLSTM
        _, output_lstm = self.convlstm(input_lstm)
        output_lstm = output_lstm[0][0]

        final_output = self.end(output_lstm)

        return final_output