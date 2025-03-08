import torch


class RandomMixUp(torch.nn.Module):
    """Randomly apply MixUp to the provided batch and targets.
    The class implements the data augmentations as described in the paper
    `"mixup: Beyond Empirical Risk Minimization" <https://arxiv.org/abs/1710.09412>`_.

    Args:
        num_classes (int): number of classes used for one-hot encoding.
        p (float): probability of the batch being transformed. Default value is 0.5.
        alpha (float): hyperparameter of the Beta distribution used for mixup.
            Default value is 1.0.
        inplace (bool): boolean to make this transform inplace. Default set to False.
    """

    def __init__(self, num_classes, p: float = 0.5, alpha: float = 1.0, inplace: bool = True) -> None:
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
            target (Tensor): Integer tensor of size (B, C, H, W, [D])

        Returns:
            Tensor: Randomly transformed batch.
        """
        was_list = False

        if isinstance(possibly_targets, (list, tuple)):
            was_list = True
        else:
            possibly_targets = [possibly_targets]

        if not self.inplace:
            batch = batch.clone()

        # It's faster to roll the batch by one instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        # Implemented as on mixup paper, page 3.
        lambda_param = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)

        mixed_targets = []

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

            # It's faster to roll the batch by one instead of shuffling it to create image pairs
            target_rolled = target.roll(1, 0)

            target_rolled.mul_(1.0 - lambda_param)
            target.mul_(lambda_param).add_(target_rolled)

            mixed_targets.append(target)

        if was_list:
            return batch, mixed_targets
        else:
            return batch, mixed_targets[0]

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}"
            f", p={self.p}"
            f", alpha={self.alpha}"
            f", inplace={self.inplace}"
            f")"
        )
        return s
