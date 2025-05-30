import warnings

from typing import Tuple, Union

from torchio.utils import to_tuple
from torchio import Subject, Resample
from torchio.typing import TypeRangeFloat
from torchio.transforms.augmentation import RandomTransform


class RandomIsotropy(RandomTransform):
    r"""Downsample an image along all axis and upsample to initial space.

    This transform simulates an image that has been acquired using isotropic
    spacing and resampled back to its original spacing.

    Similar to the work by Billot et al.: `Partial Volume Segmentation of Brain
    MRI Scans of any Resolution and
    Contrast <https://link.springer.com/chapter/10.1007/978-3-030-59728-3_18>`_.

    Args:
        axes: Axis or tuple of axes along which the image will be downsampled.
        downsampling: Downsampling factor :math:`m \gt 1`. If a tuple
            :math:`(a, b)` is provided then :math:`m \sim \mathcal{U}(a, b)`.
        image_interpolation: Image interpolation used to upsample the image
            back to its initial spacing. Downsampling is performed using
            nearest neighbor interpolation. See :ref:`Interpolation` for
            supported interpolation types.
        scalars_only: Apply only to instances of :class:`torchio.ScalarImage`.
            This is useful when the segmentation quality needs to be kept,
            as in `Billot et al. <https://link.springer.com/chapter/10.1007/978-3-030-59728-3_18>`_.
        **kwargs: See :class:`~torchio.transforms.Transform` for additional
            keyword arguments.

    Example:
        >>> import torchio as tio
        >>> transform = tio.RandomAnisotropy(axes=1, downsampling=2)
        >>> transform = tio.RandomAnisotropy(
        ...     axes=(0, 1, 2),
        ...     downsampling=(2, 5),
        ... )   # Multiply spacing of one of the 3 axes by a factor randomly chosen in [2, 5]
        >>> colin = tio.datasets.Colin27()
        >>> transformed = transform(colin)
    """  # noqa: B950

    def __init__(
            self,
            axes: Union[int, Tuple[int, ...]] = (0, 1, 2),
            downsampling: TypeRangeFloat = (1.5, 5),
            image_interpolation: str = 'linear',
            scalars_only: bool = True,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.axes = self.parse_axes(axes)
        self.downsampling_range = self._parse_range(
            downsampling,
            'downsampling',
            min_constraint=1,
        )
        parsed_interpolation = self.parse_interpolation(image_interpolation)
        self.image_interpolation = parsed_interpolation
        self.scalars_only = scalars_only

    def get_params(
            self,
            downsampling_range: Tuple[float, float],
    ) -> Tuple[int, float]:
        downsampling = self.sample_uniform(*downsampling_range)
        return downsampling

    @staticmethod
    def parse_axes(axes: Union[int, Tuple[int, ...]]):
        axes_tuple = to_tuple(axes)
        for axis in axes_tuple:
            is_int = isinstance(axis, int)
            if not is_int or axis not in (0, 1, 2):
                raise ValueError('All axes must be 0, 1 or 2')
        return axes_tuple

    def apply_transform(self, subject: Subject) -> Subject:
        is_2d = subject.get_first_image().is_2d()
        if is_2d and 2 in self.axes:
            warnings.warn(
                f'Input image is 2D, but "2" is in axes: {self.axes}',
                RuntimeWarning,
                stacklevel=2,
            )
            self.axes = list(self.axes)
            self.axes.remove(2)
        downsampling = self.get_params(
            self.downsampling_range,
        )
        target_spacing = list(subject.spacing)
        # modify all axes
        for axis in self.axes:
            target_spacing[axis] *= downsampling

        arguments = {
            'image_interpolation': 'nearest',
            'scalars_only': self.scalars_only,
        }

        sx, sy, sz = target_spacing  # for mypy
        downsample = Resample(
            target=(sx, sy, sz), **self.add_include_exclude(arguments)
        )
        downsampled = downsample(subject)
        image = subject.get_first_image()
        target = image.spatial_shape, image.affine
        upsample = Resample(
            target=target,  # type: ignore[arg-type]
            image_interpolation=self.image_interpolation,
            scalars_only=self.scalars_only,
        )
        upsampled = upsample(downsampled)
        assert isinstance(upsampled, Subject)
        return upsampled
