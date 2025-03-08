import numpy as np

from monai.transforms import RandRotateD


class RandRotatePosNeg:
    def __init__(self, range_x=(0, 0), range_y=(0, 0), range_z=(0, 0)):
        self.range_x = range_x
        self.range_y = range_y
        self.range_z = range_z

    def __call__(self, subject):
        # randonly flip the range from positive to negative
        sign_x, sign_y, sign_z = np.random.choice([-1, 1], 3)
        rotator = RandRotateD(
            keys=["image", "label"], prob=1.,
            range_x=(sign_x * self.range_x[0], sign_x * self.range_x[1]),
            range_y=(sign_y * self.range_y[0], sign_y * self.range_y[1]),
            range_z=(sign_z * self.range_z[0], sign_z * self.range_z[1]),
            mode=('bilinear', 'nearest'),
            padding_mode='zeros'
        )
        return rotator(subject)
