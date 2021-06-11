from numpy import ndim
import torch
from torch import nn
import torch.nn.functional as F


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing output of convolutions
    takes a number h_w or a tuple of (h,w) and returns a number h_w or a tuple of (h,w)

    see https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/6
    """

    if ndim(h_w) > 0 and not isinstance(kernel_size, tuple):
        kernel_size = tuple([kernel_size] * len(h_w))

    if ndim(h_w) > 0 and not isinstance(stride, tuple):
        stride = tuple([stride] * len(h_w))

    if ndim(h_w) > 0 and not isinstance(pad, tuple):
        pad = tuple([pad] * len(h_w))

    if not hasattr(h_w, '__len__'):
        h_w_prime = (h_w + (2 * pad) - (dilation * (kernel_size - 1)) - 1) // stride + 1

    else:
        h_w_prime = tuple([
            (h_w[i] + (2 * pad[i]) - (dilation * (kernel_size[i] - 1)) - 1) // stride[i] + 1
            for i in range(len(h_w))
        ])

    return h_w_prime


def conv_transpose_output_shape(h_w, kernel_size=1, stride=1, pad=0):
    """
    Utility function for computing output of transposed convolutions
    takes a number h_w or a tuple of (h,w) and returns a number h_w or a tuple of (h,w)

    see https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/6
    """

    if ndim(h_w) > 0 and not isinstance(kernel_size, tuple):
        kernel_size = tuple([kernel_size] * len(h_w))

    if ndim(h_w) > 0 and not isinstance(stride, tuple):
        stride = tuple([stride] * len(h_w))

    if ndim(h_w) > 0 and not isinstance(pad, tuple):
        pad = tuple([pad] * len(h_w))

    if not hasattr(h_w, '__len__'):
        # h_w_prime = (h_w - 1) * stride - 2 * pad + kernel_size + pad  # keras ?
        h_w_prime = (h_w - 1) * stride - 2 * pad + kernel_size          # torch

    else:
        h_w_prime = tuple([
            # (h_w[i] - 1) * stride[i] - 2 * pad[i] + kernel_size[i] + pad[i]  # keras ?
            (h_w[i] - 1) * stride[i] - 2 * pad[i] + kernel_size[i]             # torch
            for i in range(len(h_w))
        ])

    return h_w_prime


def get_layer_nd(dim, layer_prefix='Conv'):
    if len(dim) == 1:
        layer_nd = getattr(nn, f'{layer_prefix}1d')

    elif len(dim) == 2:
        layer_nd = getattr(nn, f'{layer_prefix}2d')

    elif len(dim) == 3:
        layer_nd = getattr(nn, f'{layer_prefix}3d')

    else:
        raise AssertionError("input_shape must be in (1, 2, 3)")

    return layer_nd


def get_conv_nd(dim):
    return get_layer_nd(dim, layer_prefix='Conv')


def get_conv_transpose_nd(dim):
    return get_layer_nd(dim, layer_prefix='ConvTranspose')


def get_batch_norm_nd(dim):
    return get_layer_nd(dim, layer_prefix='BatchNorm')


def get_activation_function(activation):
    try:
        return getattr(torch, activation, getattr(F, activation, None))
    except TypeError:
        return None


def call_activation(x, foo=None):
    if foo is None:
        return x

    return foo(x)
