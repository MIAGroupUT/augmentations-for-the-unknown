import os
import sys
import shutil
import subprocess

from typing import Tuple, Union, Literal

import numpy as np

from torchio import Subject
from torchio.utils import to_tuple
from torchio.transforms.augmentation import RandomTransform

from medg.transforms.additional.fastMRI import (
    create_mask_for_mask_type,
    readcfl,
    writecfl,
)


class KSpaceSubsampling(RandomTransform):
    """
    Subsample k-space data. The mask is created using a defined mask type. Reconstruction can be performed using the
    bart toolbox.
    """

    def __init__(
            self,
            mask_type: Literal['random', 'equispaced', 'magic', 'equispaced_fraction', 'magic_fraction'] = 'random',
            center_fractions: Union[float, Tuple[float, float]] = (0.08, 0.12),
            accelerations: Union[int, Tuple[int, int]] = (4, 8),
            temp_dir: str = '__temp__',
            spatial_axis: int = 3,
            **kwargs,
    ):
        """
        Args:
            mask_type: Type of mask to create. Options are 'random', 'equispaced', 'magic', 'equispaced_fraction',
                'magic_fraction'.
            center_fractions: Fraction of the center k-space to sample. If a tuple (a, b) is provided then the fraction
                is randomly chosen from a uniform distribution between a and b.
            accelerations: Acceleration factor. If a tuple (a, b) is provided then the acceleration factor is randomly
                chosen from a uniform distribution between a and b.
            **kwargs: See :class:`~torchio.transforms.Transform` for additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.mask_type = mask_type
        self.center_fractions = to_tuple(center_fractions)
        self.accelerations = to_tuple(accelerations)
        self.spatial_axis = spatial_axis

        self.temp_dir = '/'.join(['.', temp_dir])
        self.ks = '/'.join([self.temp_dir, 'ks'])
        self.sens = '/'.join([self.temp_dir, 'sens'])
        self.recon = '/'.join([self.temp_dir, 'recon'])

        # make a temp directory to store the mask
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

        os.makedirs(self.temp_dir, exist_ok=True)

        if 'win' in sys.platform:
            self.bart = ('wsl.exe', 'bart')
        else:
            self.bart = ('bart',)

    def get_params(self, shape):
        return create_mask_for_mask_type(
            mask_type_str=self.mask_type,
            center_fractions=self.center_fractions,
            accelerations=self.accelerations,
        )(shape, offset=0)[0]

    def apply_transform(self, subject: Subject) -> Subject:
        image = subject.get_first_image()
        new_data = np.zeros_like(image.data)

        for channel in range(image.data.shape[0]):
            data = image.data[channel].numpy()
            mask = self.get_params(data.shape).numpy()

            if self.spatial_axis == 2:
                kspace = np.fft.fftshift(np.fft.fft2(data, axes=(0, 1), norm="ortho"), axes=(0, 1))
            else:
                kspace = np.fft.fftshift(np.fft.fftn(data, norm="ortho"))

            kspace *= mask

            # save the k-space data using writecfl
            if self.spatial_axis == 2:
                # write each slice separately
                reconstructed_images = dict()
                for i in range(kspace.shape[-1]):
                    idx, reconstructed_image = self.run_bart_on_slice((i, kspace[..., i]))
                    reconstructed_images[idx] = reconstructed_image

                # get the reconstructed images sorted on the slice index
                reconstructed_images = [reconstructed_images[i] for i in range(kspace.shape[-1])]
                recon = np.stack(reconstructed_images, axis=-1)

                # update the image data
                new_data[channel] = recon
            else:
                writecfl(self.ks, kspace)

                # run the bart toolbox to reconstruct the image
                subprocess.run(
                    [*self.bart, 'ecalib', '-m 1', '-r 32', '-k 3', self.ks, self.sens],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )

                subprocess.run(
                    [*self.bart, 'pics', '-l1', '-r0.01', '-i10', self.ks, self.sens, self.recon],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )

                # read the reconstructed image
                recon = readcfl(self.recon)
                recon = np.abs(np.fft.ifftshift(recon))

                # update the image data
                new_data[channel] = recon

        # # delete the content of the temp directory
        # for file in os.listdir(self.temp_dir):
        #     os.remove(os.path.join(self.temp_dir, file))

        image.set_data(new_data)

        return subject

    def run_bart_on_slice(self, ikspace):
        idx, kspace = ikspace
        tempf = 'bleh'  # str(uuid.uuid4())
        os.makedirs(tempf, exist_ok=True)

        try:
            ks = '/'.join([tempf, 'ks'])
            sens = '/'.join([tempf, 'sens'])
            recon = '/'.join([tempf, 'recon'])

            writecfl(ks, kspace)
            # run the bart toolbox to reconstruct the image
            subprocess.run(
                [*self.bart, 'ecalib', '-m 1', '-r 32', '-k 3', ks, sens],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            subprocess.run(
                [*self.bart, 'pics', '-l1', '-r0.01', '-i10', ks, sens, recon],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            # read the reconstructed image
            recon = readcfl(recon)
            recon = np.abs(np.fft.ifftshift(recon))
        finally:
            pass
        #     # delete the content of the temp directory and the directory itself
        #     shutil.rmtree(tempf)

        return idx, recon


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from ml.dataset import Hippocampus

    transform = KSpaceSubsampling(
        mask_type='magic_fraction', center_fractions=(0.08, 0.12), accelerations=(4, 8), spatial_axis=2
    )
    # data_sample = tio.datasets.Colin27().t1
    data_sample = Hippocampus(
        root=os.path.join('..', '..', '..', 'data'), mode='test', split='test')[0]['image']
    print(data_sample.data.shape)
    print(data_sample.data.min(), data_sample.data.max())

    # plot the middle slice
    plt.figure()
    plt.imshow(data_sample.data[0, :, :, data_sample.shape[-1] // 2], cmap='gray')
    plt.title('Original')
    plt.show()

    # colin.plot()
    transformed = transform(data_sample)
    # transformed.plot()

    # plot the middle slice
    plt.figure()
    plt.imshow(transformed.data[0, :, :, transformed.shape[-1] // 2], cmap='gray')
    plt.title('Transformed')
    plt.show()

    print(transformed.data.shape)
    print(transformed.data.min(), transformed.data.max())
