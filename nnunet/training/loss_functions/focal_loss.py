#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import numpy as np
import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(
            self, apply_nonlin=None, alpha=0.5, gamma=2, balance_index=0, smooth=1e-5, size_average=True,
            to_onehot_y=True
    ):
        super(FocalLoss, self).__init__()
        if apply_nonlin is None:
            self.apply_nonlin = nn.Softmax(dim=1)
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average
        self.to_onehot_y = to_onehot_y

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # flatten spatial dimensions N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))

        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha
        else:
            raise TypeError(f'Unsupported alpha type: {type(alpha)}')

        alpha = alpha.to(logit.device)

        if self.to_onehot_y:
            target = torch.squeeze(target, 1)
            target = target.view(-1, 1)
            idx = target.cpu().long()

            one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
            one_hot_key = one_hot_key.scatter_(1, idx, 1)
            one_hot_key = one_hot_key.to(logit.device)

            alpha = alpha[idx]
            alpha = torch.squeeze(alpha)
        else:
            target = logit.view(target.size(0), target.size(1), -1)
            target = target.permute(0, 2, 1).contiguous()
            target = target.view(-1, target.size(-1))

            one_hot_key = target
            alpha = self.alpha * torch.ones(target.size(0), device=logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth)

        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()

        return loss


if __name__ == '__main__':
    loss = FocalLoss(
        apply_nonlin=nn.Softmax(dim=1), alpha=0.5, gamma=2, smooth=1e-5, size_average=True, to_onehot_y=False
    )
    logit = torch.randn(2, 2, 4, 4).to("cuda")
    # target = torch.randint(0, 2, (2, 1, 4, 4)).to("cuda")
    target = torch.randn(2, 2, 4, 4).to("cuda")
    print(loss(logit, target))
    print('Done')
