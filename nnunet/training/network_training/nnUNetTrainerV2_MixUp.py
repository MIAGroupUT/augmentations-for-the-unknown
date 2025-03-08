import torch

from torch.cuda.amp import autocast

from nnunet.training.data_augmentation.mixup import RandomMixUp
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet.training.network_training.nnUNetTrainerV2 import nnUNetTrainerV2
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda


class DC_and_CE_for_MixUp(DC_and_CE_loss):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ce = torch.nn.CrossEntropyLoss(**kwargs['ce_kwargs'])

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target, loss_mask=None) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        ce_loss = self.ce(net_output, target) if self.weight_ce != 0 else 0

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son")  # reserved for other stuff (later)
        return result


class nnUNetTrainerV2_MixUp(nnUNetTrainerV2):
    """
    Info for Fabian: same as internal nnUNetTrainerV2_2
    """

    def __init__(
            self, plans_file, fold, output_folder=None, dataset_directory=None, batch_dice=True, stage=None,
            unpack_data=True, deterministic=True, fp16=False
    ):
        super().__init__(
            plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data,
            deterministic, fp16
        )
        self.max_num_epochs = 300
        self.loss = DC_and_CE_for_MixUp(
            soft_dice_kwargs={'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False},
            ce_kwargs={}
        )

    def initialize(self, training=True, force_load_plans=False):
        super().initialize(training, force_load_plans)
        self.mixer = RandomMixUp(num_classes=self.num_classes, p=1.0, alpha=0.2, inplace=True)

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        """
        gradient clipping improves training stability

        :param data_generator:
        :param do_backprop:
        :param run_online_evaluation:
        :return:
        """
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = to_cuda(data)
            target = to_cuda(target)

        if do_backprop:
            data, target = self.mixer(data, target)
        else:
            new_targets = []
            if isinstance(target, (list, tuple)):
                was_list = True
            else:
                was_list = False
                target = [target]

            for _t in target:
                _t = _t.to(torch.int64)
                if self.num_classes != _t.size(1):
                    _t = _t.squeeze(1)
                    # One-hot encode the target tensor
                    if _t.ndim == 3:
                        _t = torch.nn.functional.one_hot(_t, num_classes=self.num_classes).permute(0, 3, 1, 2)
                    else:
                        _t = torch.nn.functional.one_hot(_t, num_classes=self.num_classes).permute(0, 4, 1, 2, 3)

                _t = _t.to(dtype=data.dtype)
                new_targets.append(_t)

            if was_list:
                target = new_targets
            else:
                target = new_targets[0]

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                output = self.network(data)
                del data
                l = self.loss(output, target)

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            output = self.network(data)
            del data
            l = self.loss(output, target)

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

    def run_online_evaluation(self, output, target):
        # in mixup targets are one-hot encoded, but they are expected in the original form
        # we need to convert them back to B 1 H W (D)
        if isinstance(target, (list, tuple)):
            was_list = True
        else:
            was_list = False
            target = [target]

        new_targets = []
        for _t in target:
            _t = _t.argmax(1, keepdim=True)
            new_targets.append(_t)

        if was_list:
            target = new_targets
        else:
            target = new_targets[0]

        return super().run_online_evaluation(output, target)
