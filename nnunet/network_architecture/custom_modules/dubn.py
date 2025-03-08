import torch.nn as nn


class DualBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super(DualBatchNorm2d, self).__init__()
        self.routes = ['M', 'A']

        self.bn = nn.ModuleList([nn.BatchNorm2d(num_features), nn.BatchNorm2d(num_features)])
        self.num_features = num_features
        self.ignore_model_profiling = True

        self.route = 'M'  # route images to main BN or aux BN

    def forward(self, x):
        idx = self.routes.index(self.route)
        y = self.bn[idx](x)
        return y

    @classmethod
    def convert(cls, model):
        r"""Converts all BatchNorm layers in the model to DuBN layers

        Args:
            model (nn.Module): Model to convert
        """
        old_modules = dict(model.named_children())
        for name, module in old_modules.items():
            if isinstance(module, nn.BatchNorm2d):
                setattr(model, name, DualBatchNorm2d(module.num_features))
            else:
                setattr(model, name, cls.convert(module))
        return model
