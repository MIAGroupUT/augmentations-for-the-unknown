import torch

from einops import rearrange, repeat


class AFA(torch.nn.Module):

    def __init__(
            self, img_size, min_str=0, mean_str=5, spatial_dims=2, fold_d_into_batch=False
    ):
        super().__init__()

        _x = torch.linspace(- img_size[0] / 2, img_size[0] / 2, steps=img_size[0], device='cpu')
        _y = torch.linspace(- img_size[1] / 2, img_size[1] / 2, steps=img_size[1], device='cpu')
        self._x, self._y = torch.meshgrid(_x, _y, indexing='ij')
        self.eps_scale = max(img_size) / 32

        self.spatial_dims = spatial_dims

        self.min_str = min_str
        self.mean_str = mean_str

        self.fold_d_into_batch = fold_d_into_batch

    def forward(self, x):
        init_shape = x.shape
        if len(x.shape) == 1 + self.spatial_dims:
            x = x.unsqueeze(0)

        if self.spatial_dims == 3:
            d = x.shape[-1]

        if self.spatial_dims == 3 and self.fold_d_into_batch:
            x = rearrange(x, 'b c h w d -> (b d) c h w')

        b, c, *_ = x.shape

        freqs = 1 - torch.rand((b, c, 1, 1), device=x.device)
        phases = - torch.pi * torch.rand((b, c, 1, 1), device=x.device)
        strengths = torch.empty_like(phases).exponential_(1 / self.mean_str) + self.min_str
        waves = self.gen_planar_waves(freqs, phases, x.device)

        if self.spatial_dims == 3 and (not self.fold_d_into_batch):
            # repeat the waves for each depth
            waves = repeat(waves, 'b c h w -> b c h w d', d=d)
            strengths = repeat(strengths, 'b c h w -> b c h w d', d=d)

        _temp = x + strengths * waves

        if self.spatial_dims == 3 and self.fold_d_into_batch:
            _temp = rearrange(_temp, '(b d) c h w -> b c h w d', d=d)

        return _temp.reshape(init_shape)

    def gen_planar_waves(self, freqs, phases, device):
        _x, _y = self._x.to(device), self._y.to(device)
        _waves = torch.sin(
            2 * torch.pi * freqs * (
                    _x * torch.cos(phases) + _y * torch.sin(phases)
            ) - torch.rand(1, device=device) * torch.pi
        )
        _waves.div_(_waves.norm(dim=(-2, -1), keepdim=True))

        return self.eps_scale * _waves

    def __str__(self):
        return f'AFA(' \
               f'min_str={self.min_str}, mean_str={self.mean_str}' \
               f')'


class ComplexAFA(torch.nn.Module):

    def __init__(
            self, img_size, mean_str=5
    ):
        super().__init__()

        self.img_size = img_size
        self.mean_str = mean_str

        self.eps_scale = img_size / 32

    def forward(self, x):
        init_shape = x.shape
        if len(x.shape) < 4:
            x = rearrange(x, 'c h w -> () c h w')
        b, c, h, w = x.shape

        aug = torch.empty((b, c, h, w // 2, 2), device=x.device).uniform_(-1, 1)
        aug[..., 0] = aug[..., 0].sign()
        aug[..., 1] *= torch.pi

        drop = torch.rand((b, c, h, w // 2, 1), device=x.device) < torch.empty((b, c, 1, 1, 1),
                                                                               device=x.device).uniform_(
            self.lower_bound, 1.)
        aug.masked_fill_(drop, 0)

        aug = torch.fft.irfft2(torch.view_as_complex(aug), s=(h, w), dim=(2, 3), norm='ortho').real
        aug.div_(aug.norm(dim=(-2, -1), keepdim=True) + 1e-6)

        strengths = self.eps_scale * torch.empty((b, c, 1, 1), device=x.device).exponential_(1 / self.mean_str)

        return torch.clamp(x + strengths * aug, 0, 1).reshape(init_shape)

    def __str__(self):
        return f'ComplexAFA(' \
               f'image_size={self.img_size}, ' \
               f'mean_str={self.mean_str}' \
               f')'


if __name__ == '__main__':
    import os

    import matplotlib.pyplot as plt

    from monai.data import ThreadDataLoader

    from ml.dataset.acdc import ACDC

    ds = ACDC(root=os.path.join('..', '..', '..', 'data'), mode='train', split='train', split_seed=None, cache_rate=0.1)
    dl = ThreadDataLoader(ds, num_workers=0, batch_size=1)

    afa = AFA(180, min_str=0, mean_str=20, spatial_dims=3, fold_d_into_batch=True)

    for i, batch in enumerate(dl):
        if i == 5:
            break

        x, y = batch['image'], batch['label']
        afa_x = afa(x)

        # plot 3 slices and two columns showing the original and augmented images
        fig, axs = plt.subplots(3, 2, figsize=(8, 12), sharex=True, sharey=True)
        depth = x.shape[-1]
        inc = depth // 3

        for j in range(3):
            axs[j][0].imshow(x[0][0][..., inc * j], cmap='gray')
            axs[j][0].imshow(y[0][0][..., inc * j], alpha=0.5)
            axs[j][0].set_title('Original')

        for j in range(3):
            axs[j][1].imshow(afa_x[0][0][..., inc * j], cmap='gray')
            axs[j][1].imshow(y[0][0][..., inc * j], alpha=0.5)
            axs[j][1].set_title('AFA')

        plt.tight_layout()
        plt.show()
