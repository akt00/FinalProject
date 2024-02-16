import os
# external libs
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset
from torch import Tensor


def data_pipeline(train: bool) -> A.Compose:
    """ feature engineering pipeline """
    if train:
        pipeline = A.Compose([
            A.ColorJitter(),
            A.GaussNoise(),
            A.RandomRotate90(),
            A.Flip(),
            ToTensorV2(True),
        ])
    else:  # data pipeline for evaluation
        pipeline = A.Compose([
            ToTensorV2(True),
        ])
    return pipeline


class BrainDatasetv2(Dataset):
    """ Brain Tumor data loder version 2

    Upgraded the previous implementation of BrainDataloader to support \
    oversampling and complex data augmentation pipeline

    Attributes:
        train: loads train dataset if true, load test dataset otherwise
        data_transform: data pipeline for preprocessing and data agumentation
    """
    def __init__(self, train: bool, data_transform: A.Compose,
                 oversample: bool) -> None:
        super().__init__()
        self.train = train
        self.data_transform = data_transform
        # dir path for the train/test dataset
        data_dir = 'train' if train else 'test'
        data_dir = os.path.abspath(os.path.join('dataset', data_dir))
        assert os.path.exists(data_dir), './dataset/[train|test]'
        self.images: list[str] = []
        self.masks: list[str] = []
        # load image/mask paths
        for data in os.listdir(data_dir):
            if data.find('mask') >= 0:
                mask_path = os.path.join(data_dir, data)
                self.masks.append(mask_path)
                assert os.path.exists(mask_path)
                img_path = ''.join(mask_path.split('_mask'))
                self.images.append(img_path)
        # only oversample brain tumor data points
        if oversample:
            self.oversample()
        assert len(self.images) == len(self.masks)

    def oversample(self) -> None:
        """ oversamples images that contains tumors """
        _images, _masks = [], []  # temporary lists for performance
        for path in self.masks:
            mask = cv.imread(path, cv.IMREAD_GRAYSCALE)
            # brain tumor exists
            if np.any(mask == 255.):
                im_path = ''.join(path.split('_mask'))
                assert os.path.exists(im_path) and os.path.exists(path), \
                    'Invalid path. Make sure the absolute path does not \
                     contain "_mask" in the path.'
                _images.append(im_path)
                _masks.append(path)
        self.images += _images
        self.masks += _masks

    def __len__(self) -> int:
        """ returns the dataset size """
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """ returns the image/mask pair after preprocessing & data aug """
        image = cv.imread(self.images[index])
        mask = cv.imread(self.masks[index], cv.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, axis=-1)
        assert image.dtype == np.uint8 and mask.dtype == np.uint8
        transformed = self.data_transform(image=image, mask=mask)
        image, mask = transformed['image'], transformed['mask']
        # normalize to 0-1
        return image / 255, mask / 255
