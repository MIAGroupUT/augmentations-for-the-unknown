import torch
import torch.nn as nn

from .dubn import DualBatchNorm2d


class DuBIN(nn.Module):
    r"""Instance-Batch Normalization layer from
    `"Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net"
    <https://arxiv.org/pdf/1807.09441.pdf>`

    Args:
        planes (int): Number of channels for the input tensor
        ratio (float): Ratio of instance normalization in the IBN layer
    """

    def __init__(self, planes):
        super(DuBIN, self).__init__()
        self.half = int(planes * 0.5)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = DualBatchNorm2d(planes - self.half)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out

    @classmethod
    def convert(cls, model):
        r"""Converts all BatchNorm layers in the model to DuBIN layers

        Args:
            model (nn.Module): Model to convert
        """
        old_modules = dict(model.named_children())
        for name, module in old_modules.items():
            if isinstance(module, nn.BatchNorm2d):
                setattr(model, name, DuBIN(module.num_features))
            else:
                setattr(model, name, cls.convert(module))
        return model
