from monai.config import KeysCollection
from monai.transforms import ResizeWithPadOrCrop
from torchio import Subject


class TrackOriginalSized:
    """
    Track the original size of the image.
    """

    def __init__(self, keys: KeysCollection, original_shape_key: str):
        self.keys = keys
        self.original_shape_key = original_shape_key

    def __call__(self, subject: Subject) -> Subject:
        for key in self.keys:
            image = subject[key]
            subject[self.original_shape_key] = image.shape
        return subject


class ResizeToOriginald:
    """
    Resize the image to the original size.
    """

    def __init__(self, keys: KeysCollection, original_shape_key: str):
        self.keys = keys
        self.original_shape_key = original_shape_key

    def __call__(self, subject: Subject) -> Subject:
        for key in self.keys:
            image = subject[key]
            original_shape = subject[self.original_shape_key]
            spatial_size = original_shape[1:]

            subject[key] = ResizeWithPadOrCrop(spatial_size=spatial_size, mode='minimum')(image)
        return subject
