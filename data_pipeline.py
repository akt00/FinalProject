import cv2 as cv
import os
from torch.utils.data import Dataset


# the code in this section is originally written by Akihiro Tanimoto
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
# the cell ends here


# the code in this section is originally written by Akihiro Tanimoto
if __name__ == '__main__':
    # testing the trained model
    import torch
    from torchvision import transforms
    from losses import DiceLoss

    class RescaleMask:
        def __call__(self, img):
            return img.clamp(min=0, max=1)

    mask_transform = transforms.Compose([
        transforms.ToTensor(),
        RescaleMask()
    ])
    train_loader = BrainDataset(True, transforms.ToTensor(), mask_transform)
    test_loader = BrainDataset(False, transforms.ToTensor(), mask_transform)
    print(len(train_loader), len(test_loader))
    model = torch.load('models/model.pth')
    device = torch.device('cuda')
    model.eval().to(device)
    loss_fn = DiceLoss().to(device)

    for img, gt in test_loader:
        img, gt = img.to(device), gt.to(device)
        img = img.unsqueeze(0)
        pred = model(img.to(device)).squeeze(0)
        print(f'loss: {loss_fn(pred, gt).item():.7f}')
        img = img.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        gt = gt.detach().cpu().numpy().transpose(1, 2, 0)
        pred = (pred > .9).float().detach().cpu().numpy()\
            .transpose(1, 2, 0) * 255
        cv.imshow('Input Image', img)
        cv.imshow('Model Prediction', pred)
        cv.imshow('Ground Truth', gt)
        cv.waitKey(400)
# the cell ends here
