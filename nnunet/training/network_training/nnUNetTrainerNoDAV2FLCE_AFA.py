import torch
from torch.cuda.amp import autocast

from nnunet.training.data_augmentation.afa import AFA
from nnunet.network_architecture.custom_modules.dubin import DuBIN
from nnunet.training.network_training.nnUNetTrainerNoDAV2FLCE import nnUNetTrainerNoDAV2FLCE

from nnunet.utilities.to_torch import maybe_to_torch, to_cuda


class nnUNetTrainerNoDAV2FLCE_AFA(nnUNetTrainerNoDAV2FLCE):
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
        self.max_num_epochs = 200

    def initialize(self, training=True, force_load_plans=False):
        super().initialize(training, force_load_plans)
        self.afa = AFA(self.patch_size, min_str=0, mean_str=10, spatial_dims=2, fold_d_into_batch=False)
        self.network = DuBIN.convert(self.network)

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

        self.optimizer.zero_grad()

        if self.fp16:
            with autocast():
                self.network.apply(lambda m: setattr(m, 'route', 'M'))
                output = self.network(data)
                if do_backprop:
                    self.network.apply(lambda m: setattr(m, 'route', 'A'))
                    output_aux = self.network(self.afa(data))
                    self.network.apply(lambda m: setattr(m, 'route', 'M'))
                    l = 0.5 * ((clean_loss := self.loss(output, target)) + self.loss(output_aux, target))
                else:
                    l = clean_loss = self.loss(output, target)
                del data

            if do_backprop:
                self.amp_grad_scaler.scale(l).backward()
                self.amp_grad_scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.amp_grad_scaler.step(self.optimizer)
                self.amp_grad_scaler.update()
        else:
            self.network.apply(lambda m: setattr(m, 'route', 'M'))
            output = self.network(data)
            if do_backprop:
                self.network.apply(lambda m: setattr(m, 'route', 'A'))
                output_aux = self.network(self.afa(data))
                self.network.apply(lambda m: setattr(m, 'route', 'M'))
                l = 0.5 * ((clean_loss := self.loss(output, target)) + self.loss(output_aux, target))
            else:
                l = clean_loss = self.loss(output, target)
            del data

            if do_backprop:
                l.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
                self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return clean_loss.detach().cpu().numpy()
