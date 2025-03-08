import torch
import numpy as np
import torch.nn as nn

from einops import rearrange
from albumentations import PadIfNeeded

from nnunet.network_architecture.custom_modules.dubin import DuBIN

from .generic_unet import GenericUNet


def process_plans_and_init(plans_dict, checkpoint_path=None, deep_supervision=True, return_latent=False):
    """
    This function processes the plans dictionary and returns the relevant keys for the model initialization
    :param plans_dict: dictionary containing the plans for the model
    :param checkpoint_path: path to the checkpoint to load. If None, the model will be initialized from scratch
    :param deep_supervision: whether to use deep supervision
    :param return_latent: whether to return the latent space
    :return: a list containing the relevant keys for the model initialization
    """
    patch_size = plans_dict['plans_per_stage'][0]['patch_size']
    is_three_3d = len(patch_size) == 3

    if is_three_3d:
        conv_op = nn.Conv3d
        norm_op = nn.InstanceNorm3d
        dropout_op = nn.Dropout3d
    else:
        conv_op = nn.Conv2d
        norm_op = nn.InstanceNorm2d
        dropout_op = nn.Dropout2d

    kwargs = {
        'input_channels': plans_dict['num_modalities'],
        'base_num_features': plans_dict['base_num_features'],
        'num_classes': plans_dict['num_classes'] + 1,
        'num_pool': len(plans_dict['plans_per_stage'][0]['pool_op_kernel_sizes']),
        'num_conv_per_stage': plans_dict.get('conv_per_stage', 2),
        'feat_map_mul_on_downscale': plans_dict.get('feat_map_mul_on_downscale', 2),
        'conv_op': conv_op,
        'norm_op': norm_op,
        'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
        'dropout_op': dropout_op,
        'dropout_op_kwargs': {'p': 0, 'inplace': True},
        'nonlin': nn.LeakyReLU,
        'nonlin_kwargs': {'negative_slope': 1e-2, 'inplace': True},
        'deep_supervision': deep_supervision,
        'dropout_in_localization': False,
        'pool_op_kernel_sizes': plans_dict['plans_per_stage'][0]['pool_op_kernel_sizes'],
        'conv_kernel_sizes': plans_dict['plans_per_stage'][0]['conv_kernel_sizes'],
        'upscale_logits': False,
        'convolutional_pooling': True,
        'convolutional_upsampling': True,
        'final_nonlin': lambda x: x,
    }

    model = GenericUNet(**kwargs, return_latent=return_latent)
    if checkpoint_path is not None:
        if 'AFA' in str(checkpoint_path):
            model = DuBIN.convert(model)
        weights_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(weights_dict['state_dict'])
    model.eval()

    return model


def prepare_scan(scan, label, num_classes=3):
    # normalise the scan, each channel using the mean and std of the scan itself
    # the scan should have shape (c, h, w, d)
    scan = (scan - scan.mean(axis=(1, 2, 3), keepdims=True)) / (scan.std(axis=(1, 2, 3), keepdims=True) + 1e-8)
    label = np.eye(num_classes)[np.round(label).astype(int)].transpose(3, 0, 1, 2)

    # change the shape such that depth is the first dimension (pseudo-batch)
    scan = rearrange(scan, 'c h w d -> d h w c')
    label = rearrange(label, 'c h w d -> d h w c')

    original_shape = scan.shape

    # pad the scan such that height and width are divisible by 32
    padder = PadIfNeeded(
        min_height=None, min_width=None, pad_height_divisor=32, pad_width_divisor=32,
        border_mode=0, value=0, always_apply=True, p=1.0
    )
    min_dim = PadIfNeeded(
        min_height=256, min_width=256, pad_height_divisor=None, pad_width_divisor=None,
        border_mode=0, value=0, always_apply=True, p=1.0
    )
    _padded_slices = []
    _padded_labels = []
    for depth in range(scan.shape[0]):
        padded = padder(image=scan[depth], mask=label[depth])
        padded = min_dim(image=padded['image'], mask=padded['mask'])
        _padded_slices.append(padded['image'])
        _padded_labels.append(padded['mask'])

    scan = np.stack(_padded_slices, axis=0)
    label = np.stack(_padded_labels, axis=0)

    scan = rearrange(scan, 'd h w c -> d c h w')
    label = rearrange(label, 'd h w c -> d c h w')

    # convert to tensor and move to GPU
    scan = torch.from_numpy(scan).float().cuda()
    label = torch.from_numpy(label).float().cuda()

    return scan, label, original_shape


def crop_to_original_shape(scan, original_shape):
    # crop the scan to the original shape
    # the scan should have shape (d, c, h, w)
    # the original shape of h, w is in the middle of the scan
    h_pad_amount = scan.shape[2] - original_shape[1]
    w_pad_amount = scan.shape[3] - original_shape[2]

    top_h_pad = h_pad_amount // 2
    left_w_pad = w_pad_amount // 2

    scan = scan[..., top_h_pad:original_shape[1] + top_h_pad, left_w_pad:original_shape[2] + left_w_pad]

    return scan
