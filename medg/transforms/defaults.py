# Copyright (C) 2022  AICONS Lab

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import torchio as tio
import monai.transforms as tf

from medg.transforms.additional import (
    RandomIsotropy  # my implementation of IsotropicDownsampling as original implementation was not available
)

from medg.transforms.additional.defaults import DEFAULT_ADDITIONAL_TRANSFORM_SETTINGS

DEFAULT_TRANSFORM_SETTINGS = {
    'AnisoDownsample': {
        'transform': tio.RandomAnisotropy(
            include=['image'],
            axes=(0, 1,)
        ),
        'pre_transforms': [
            tf.ToTensord(keys=['image', 'label'])
        ],
        'post_transforms': [],
        'severity_controller': {
            'downsampling_range': [(x,) * 2 for x in np.linspace(1., 10., 6)[1:]]
        }
    },
    'BiasField': {
        'transform': tio.transforms.RandomBiasField(include=['image']),
        'pre_transforms': [
            tf.ToTensord(keys=['image', 'label'])
        ],
        'post_transforms': [],
        'severity_controller': {
            'coefficients_range': [(-x, x) for x in np.linspace(0., 1.5, 6)[1:]]
        }
    },
    'ContrastCompression': {
        'transform': tf.AdjustContrastd(keys=['image'], gamma=1.0),
        'pre_transforms': [],
        'post_transforms': [
            tf.ToTensord(keys=['image', 'label'])
        ],
        'severity_controller': {
            'adjuster.gamma': np.linspace(1., 0.3, 6)[1:]
        }
    },
    'ContrastExpansion': {
        'transform': tf.AdjustContrastd(keys=['image'], gamma=1.0),
        'pre_transforms': [],
        'post_transforms': [
            tf.ToTensord(keys=['image', 'label'])
        ],
        'severity_controller': {
            'adjuster.gamma': np.linspace(1., 3., 6)[1:]
        }
    },
    'ElasticDeformation': {
        'transform': tio.transforms.RandomElasticDeformation(include=['image', 'label']),
        'pre_transforms': [
            tf.ToTensord(keys=['image', 'label'])
        ],
        'post_transforms': [],
        'severity_controller': {
            'max_displacement': [(x, x, 0) for x in np.linspace(0., 30., 6)[1:]]
        }
    },
    'Ghosting': {
        'transform': tio.transforms.RandomGhosting(
            include=['image'],
            axes=(0, 1)
        ),
        'pre_transforms': [
            tf.ToTensord(keys=['image', 'label'])
        ],
        'post_transforms': [
            tf.ToNumpyd(keys=['image']),
            tf.ThresholdIntensityd(keys=['image'], threshold=0.0, above=True, cval=0.0),
            tf.ToTensord(keys=['image'])
        ],
        'severity_controller': {
            'num_ghosts_range': [(x,) * 2 for x in [3, 5, 7, 9, 11]],
            'intensity_range': [(x,) * 2 for x in np.linspace(0.0, 2.5, 6)[1:]]
        }
    },
    'IsoDownsample': {
        'transform': RandomIsotropy(
            include=['image', ],
            axes=(0, 1)
        ),
        'pre_transforms': [
            tf.ToTensord(keys=['image', 'label'])
        ],
        'post_transforms': [],
        'severity_controller': {
            'downsampling_range': [(x,) * 2 for x in np.linspace(1., 4., 6)[1:]]
        }
    },
    'RandomMotion': {
        'transform': tio.transforms.RandomMotion(include=['image']),
        'pre_transforms': [
            tf.ToTensord(keys=['image', 'label'])
        ],
        'post_transforms': [
            tf.ToNumpyd(keys=['image']),
            tf.ThresholdIntensityd(keys=['image'], threshold=0.0, above=True, cval=0.0),
            tf.ToTensord(keys=['image'])
        ],
        'severity_controller': {
            'degrees_range': [(-x, x) for x in np.linspace(0.0, 5.0, 6)[1:]],
            'translation_range': [(-x, x) for x in np.linspace(0.0, 10.0, 6)[1:]],
            'num_transforms': [2, 4, 6, 8, 10]
        }
    },
    'RicianNoise': {
        'transform': tf.RandRicianNoised(keys=['image'], prob=1.0, channel_wise=True, relative=True, sample_std=False),
        'pre_transforms': [],
        'post_transforms': [
            tf.ToTensord(keys=['image', 'label'])
        ],
        'severity_controller': {
            'rand_rician_noise.std': np.linspace(0., 0.8, 6)[1:]
        }
    },
    'Smoothing': {
        'transform': tio.transforms.Blur(include=['image'], std=(0.0,) * 3),
        'pre_transforms': [
            tf.ToTensord(keys=['image', 'label'])
        ],
        'post_transforms': [],
        'severity_controller': {
            'std': [(x,) * 3 for x in np.linspace(0., 4., 6)[1:]]
        }
    },
    **DEFAULT_ADDITIONAL_TRANSFORM_SETTINGS
}
