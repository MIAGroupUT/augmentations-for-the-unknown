import math

import torch


class RandomCutMix2D(torch.nn.Module):
    """Randomly apply CutMix to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features"
    <https://arxiv.org/abs/1905.04899>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for cutmix.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")

        self.p = p
        self.num_classes = num_classes
        self.alpha = alpha
        self.inplace = inplace

    @torch.no_grad()
    def forward(self, batch, possibly_targets):
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W, [D])
            possibly_targets (Tensor): Integer tensor of size (B, C, [H, W], [D])

        Returns:
            Tensor: Randomly transformed batch.
        """
        if torch.rand(1).item() >= self.p:
            do_aug = False
        else:
            do_aug = True

        was_list = False

        if isinstance(possibly_targets, (list, tuple)):
            was_list = True
        else:
            possibly_targets = [possibly_targets]

        cutmixed_targets = []

        if not self.inplace:
            batch = batch.clone()

        batch_rolled = batch.roll(1, 0)

        # Implemented as on cutmix paper, page 12 (with minor corrections on typos).
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        _, _, H, W = batch.shape

        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        if do_aug:
            batch[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]

        for target in possibly_targets:

            if not self.inplace:
                target = target.clone()

            target = target.to(torch.int64)

            if self.num_classes != target.size(1):
                target = target.squeeze(1)
                # One-hot encode the target tensor
                if target.ndim == 3:
                    target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2)
                else:
                    target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).permute(0, 4, 1, 2, 3)

            target = target.to(dtype=batch.dtype)

            target_rolled = target.roll(1, 0)

            if target.ndim > 2:
                # then the target is also spatial
                # we need to scale the x1, x2, y1, y2 values to the target size
                scaling_ratio_x, scaling_ratio_y = target.shape[-1] / W, target.shape[-2] / H
                t_x1, t_x2, t_y1, t_y2 = int(x1 * scaling_ratio_x), int(x2 * scaling_ratio_x), int(
                    y1 * scaling_ratio_y), int(y2 * scaling_ratio_y)
                if (t_x1 - t_x2) * (t_y1 - t_y2) == 0:
                    cutmixed_targets.append(target)
                else:
                    if do_aug:
                        target[:, :, t_y1:t_y2, t_x1:t_x2] = target_rolled[:, :, t_y1:t_y2, t_x1:t_x2]
                    cutmixed_targets.append(target)
            else:
                lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

                target = target.to(batch.dtype)

                target_rolled.mul_(1.0 - lambda_param)
                target.mul_(lambda_param).add_(target_rolled)

                cutmixed_targets.append(target)

        if was_list:
            return batch, cutmixed_targets

        return batch, cutmixed_targets[0]

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s

    def __str__(self) -> str:
        return self.__repr__()


class RandomCutMix3D(torch.nn.Module):

    def __init__(self, num_classes: int, p: float = 0.5, alpha: float = 1.0, inplace: bool = False) -> None:
        super().__init__()
        if alpha <= 0:
            raise ValueError("Alpha param can't be zero.")

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch, possibly_tagets):
        """
        Args:
            batch (Tensor): Float tensor of size (B, C, H, W, [D])
            target (Tensor): Integer tensor of size (B, C, [H, W], [D])

        Returns:
            Tensor: Randomly transformed batch.
        """

        if torch.rand(1).item() >= self.p:
            do_aug = False
        else:
            do_aug = True

        was_list = False

        if isinstance(possibly_tagets, (list, tuple)):
            was_list = True
        else:
            possibly_tagets = [possibly_tagets]

        cutmixed_targets = []

        if not self.inplace:
            batch = batch.clone()

        batch_rolled = batch.roll(1, 0)

        # Implemented as on cutmix paper, page 12 (with minor corrections on typos).
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha, self.alpha]))[0])
        _, _, H, W, D = batch.shape

        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))
        r_z = torch.randint(D, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)
        r_d_half = int(r * D)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        z1 = int(torch.clamp(r_z - r_d_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))
        z2 = int(torch.clamp(r_z + r_d_half, max=D))

        if do_aug:
            batch[:, :, y1:y2, x1:x2, z1:z2] = batch_rolled[:, :, y1:y2, x1:x2, z1:z2]

        for target in possibly_tagets:

            if not self.inplace:
                target = target.clone()

            target = target.to(torch.int64)

            if self.num_classes != target.size(1):
                target = target.squeeze(1)
                # One-hot encode the target tensor
                if target.ndim == 3:
                    target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2)
                else:
                    target = torch.nn.functional.one_hot(target, num_classes=self.num_classes).permute(0, 4, 1, 2, 3)

            target = target.to(dtype=batch.dtype)

            target_rolled = target.roll(1, 0)

            if target.ndim > 2:
                # then the target is also spatial
                # we need to scale the x1, x2, y1, y2 values to the target size
                scaling_ratio_x, scaling_ratio_y, scaling_ratio_z = target.shape[-2] / W, target.shape[-3] / H, \
                                                                    target.shape[-1] / D

                t_x1, t_x2, t_y1, t_y2, t_z1, t_z2 = int(x1 * scaling_ratio_x), int(x2 * scaling_ratio_x), int(
                    y1 * scaling_ratio_y), int(y2 * scaling_ratio_y), int(z1 * scaling_ratio_z), int(
                    z2 * scaling_ratio_z)
                if (t_x1 - t_x2) * (t_y1 - t_y2) * (t_z1 - t_z2) == 0:
                    cutmixed_targets.append(target)
                else:
                    if do_aug:
                        target[:, :, t_y1:t_y2, t_x1:t_x2, t_z1:t_z2] = target_rolled[:, :, t_y1:t_y2, t_x1:t_x2, t_z1:t_z2]
                    cutmixed_targets.append(target)
            else:
                lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) * (z2 - z1) / (W * H * D))

                target = target.to(batch.dtype)
                target_rolled = target.roll(1, 0).to(batch.dtype)

                target_rolled.mul_(1.0 - lambda_param)
                target.mul_(lambda_param).add_(target_rolled)

                cutmixed_targets.append(target)

        if was_list:
            return batch, cutmixed_targets

        return batch, cutmixed_targets[0]


if __name__ == '__main__':
    # try both 2D and 3D
    import torch
    import matplotlib.pyplot as plt

    # 2D
    num_classes = 2
    p = 1.0
    alpha = 1.0
    inplace = False

    cutmix2d = RandomCutMix2D(num_classes, p=p, alpha=alpha, inplace=inplace)
    batch = torch.zeros(2, 3, 32, 32)
    batch[0, :, 10:20, 10:20] = 1.0
    batch[1, :, 5:15, 5:15] = 0.5
    target = torch.zeros(2, num_classes, 32, 32, dtype=torch.int64)
    target[0, 1, 10:20, 10:20] = 1
    target[1, 1, 5:15, 5:15] = 1
    print(cutmix2d)
    cutmixed_batch = cutmix2d(batch, target)
    print(cutmixed_batch[0].shape, cutmixed_batch[1].shape)

    for i in range(2):
        _, axs = plt.subplots(2, 2)
        axs = axs.flatten()
        axs[0].imshow(batch[i, 0].detach().numpy())
        axs[0].set_title('Original')
        axs[1].imshow(cutmixed_batch[0][i, 0].detach().numpy())
        axs[1].set_title('CutMixed')
        axs[2].imshow(target[i].detach().numpy().argmax(0) + 1, vmin=0, vmax=num_classes)
        axs[3].imshow(cutmixed_batch[1][i].detach().numpy().argmax(0) + 1, vmin=0, vmax=num_classes)
        plt.show()

    # 3D
    num_classes = 2
    p = 1.

    cutmix3d = RandomCutMix3D(num_classes, p=p, alpha=alpha, inplace=inplace)
    batch = torch.zeros(2, 3, 32, 32, 32)
    batch[0, :, 10:20, 10:20, 10:20] = 1.0
    batch[1, :, 5:15, 5:15, 5:15] = 0.5
    target = torch.zeros(2, num_classes, 32, 32, 32, dtype=torch.int64)
    target[0, 1, 10:20, 10:20, 10:20] = 1
    target[1, 1, 5:15, 5:15, 5:15] = 1
    print(cutmix3d)

    cutmixed_batch = cutmix3d(batch, target)
    print(cutmixed_batch[0].shape, cutmixed_batch[1].shape)

    for i in range(2):
        _, axs = plt.subplots(2, 2)
        axs = axs.flatten()
        axs[0].imshow(batch[i, 0, 16].detach().numpy())
        axs[0].set_title('Original')
        axs[1].imshow(cutmixed_batch[0][i, 0, 16].detach().numpy())
        axs[1].set_title('CutMixed')
        axs[2].imshow(target[i].detach().numpy().argmax(0)[16] + 1, vmin=0, vmax=num_classes)
        axs[3].imshow(cutmixed_batch[1][i].detach().numpy().argmax(0)[16] + 1, vmin=0, vmax=num_classes)
        plt.show()
