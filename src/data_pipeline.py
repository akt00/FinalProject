import os
import warnings
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
            A.MedianBlur(3, always_apply=True),
            A.OneOf([
                A.Sequential([
                    A.Flip(p=0.5),
                    A.Rotate(180, border_mode=0, always_apply=True),
                ], p=0.5),
                A.ElasticTransform(border_mode=0, p=0.5),
            ], p=0.5),
            ToTensorV2(True),
        ])
    else:  # data pipeline for evaluation
        pipeline = A.Compose([
            A.MedianBlur(3, always_apply=True),
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
        assert os.path.exists(data_dir), 'dataset directory must follow \
            ./dataset/[train|test]'
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
        # only oversample on train dataset
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


loader = BrainDatasetv2(True, data_pipeline(False), oversample=False)
for img, mask in loader:
    print(img.dtype, mask.dtype)
    break


class BrainDataset(Dataset):
    """ Extention of torch's Dataset class

    Attributes:
        train (bool): true if dataset is for trainig, false for testing
        img_transform: callable applied to loaded images in numpy
        target_transform: callable applied to the target label
    """
    def __init__(self, train=True, img_transform=None,
                 target_transform=None) -> None:
        super().__init__()
        warnings.warn('This data loader is deprecated', DeprecationWarning)
        self.train = train
        self.img_transform = img_transform
        self.target_transform = target_transform
        dirs = [os.path.abspath(os.path.join('dataset', d))
                for d in os.listdir('dataset')
                if os.path.isdir(os.path.join('dataset', d))]
        # sort dirs to make sure directories are always in the same order
        dirs.sort()
        assert len(dirs) == 110, f'Error in the dataset directory, {len(dirs)}'

        train_dirs = dirs[:int(len(dirs) * .8)]
        assert len(train_dirs) == 88, 'train size must be 88'
        test_dirs = dirs[int(len(dirs) * .8):]
        assert len(test_dirs) == 22, 'test size must be 22'
        # constructs train dataset
        if train:
            self.train_images, self.train_masks = [], []
            for dir in train_dirs:
                for file in os.listdir(dir):
                    if file.find('mask') >= 0:
                        label_path = os.path.join(dir, file)
                        file = ''.join(file.split('_mask'))
                        im_path = os.path.join(dir, file)
                        assert os.path.exists(im_path)
                        assert os.path.exists(label_path)
                        self.train_images.append(im_path)
                        self.train_masks.append(label_path)
            assert len(self.train_images) == len(self.train_masks)
        # constructs test dataset
        else:
            self.test_images, self.test_masks = [], []
            for dir in test_dirs:
                for file in os.listdir(dir):
                    if file.find('mask') >= 0:
                        label_path = os.path.join(dir, file)
                        file = ''.join(file.split('_mask'))
                        im_path = os.path.join(dir, file)
                        assert os.path.exists(im_path)
                        assert os.path.exists(label_path)
                        self.test_images.append(im_path)
                        self.test_masks.append(label_path)
            assert len(self.test_images) == len(self.test_masks)

    def __len__(self):
        """ returns the dataset size """
        if self.train:
            return len(self.train_images)
        else:
            return len(self.test_images)

    def __getitem__(self, index) -> any:
        """ returns the loaded (img, label) at index """
        if self.train:
            image = cv.imread(self.train_images[index])
            label = cv.imread(self.train_masks[index], cv.IMREAD_GRAYSCALE)
            if self.img_transform:
                image = self.img_transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return image, label
        else:
            image = cv.imread(self.test_images[index])
            label = cv.imread(self.test_masks[index], cv.IMREAD_GRAYSCALE)
            if self.img_transform:
                image = self.img_transform(image)
            if self.target_transform:
                label = self.target_transform(label)
            return image, label
