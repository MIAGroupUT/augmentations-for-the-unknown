import numpy as np
import monai.transforms as tf

from medg.transforms.additional import KSpaceSubsampling, RandomSpikeN
from medg.utils import TrackOriginalSized, ResizeToOriginald, RandRotatePosNeg

DEFAULT_ADDITIONAL_TRANSFORM_SETTINGS = {
    'Rotation': {
        'transform': RandRotatePosNeg(),
        'pre_transforms': [
            tf.Orientationd(keys=["image", "label"], axcodes="RAS"),
            TrackOriginalSized(keys=['image', 'label'], original_shape_key='original_shape'),
            tf.CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            tf.ToTensord(keys=['image', 'label'], track_meta=True),
        ],
        'post_transforms': [
            ResizeToOriginald(keys=['image', 'label'], original_shape_key='original_shape'),
            tf.ToTensord(keys=['image', 'label']),
        ],
        'severity_controller': {
            'range_z': [
                (l_r, u_r)
                for (l_r, u_r) in map(
                    lambda d: ((d - 6) * np.pi / 180., d * np.pi / 180.),
                    np.linspace(0., 30., 6)[1:]
                )
            ],
            'range_x': [
                (0, 0) for _ in np.linspace(0., 30., 6)[1:]
            ],
            'range_y': [
                (0, 0) for _ in np.linspace(0., 30., 6)[1:]
            ]
        }
    },
    'Scale': {
        'transform': tf.RandAffineD(
            keys=["image", "label"], prob=1.,
            mode=('bilinear', 'nearest'),
            padding_mode='zeros'
        ),
        'pre_transforms': [
            TrackOriginalSized(keys=['image', 'label'], original_shape_key='original_shape'),
            tf.CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
            tf.ToTensord(keys=['image', 'label'], track_meta=True),
        ],
        'post_transforms': [
            ResizeToOriginald(keys=['image', 'label'], original_shape_key='original_shape'),
            tf.ToTensord(keys=['image', 'label']),
        ],
        'severity_controller': {
            'rand_affine.rand_affine_grid.scale_range': [(-x / 2, x) for x in np.linspace(0., 1., 6)[1:]]
        }
    },
    'SpikeNoise': {
        'transform': RandomSpikeN(include=['image'], num_spikes=(1, 3), intensity=(1, 3), spatial_axis=2),
        'pre_transforms': [
            tf.ToTensord(keys=['image', 'label'])
        ],
        'post_transforms': [],
        'severity_controller': {
            'num_spikes_range': [(1, 1), (1, 1), (2, 2), (2, 2), (3, 3)],
            'intensity_range': [(1, 1.5), (1.5, 3.), (1, 1.5), (1.5, 3.), (1, 2)]
        }
    },
    'KSpaceSubsampling': {
        'transform': KSpaceSubsampling(mask_type='magic_fraction', include=['image'], spatial_axis=2),
        'pre_transforms': [
            tf.ToTensord(keys=['image', 'label']),
        ],
        'post_transforms': [],
        'severity_controller': {
            'center_fractions': [(0.16,), (0.16,), (0.08,), (0.08,), (0.04,)],
            'accelerations': [(4,), (8,), (4,), (8,), (4,)],
        }
    },
}
