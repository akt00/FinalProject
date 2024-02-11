# 3rd party libs
import cv2 as cv
import torch
from torchvision.transforms import ToTensor
# local modules
from src.data_pipeline import BrainDataset


if __name__ == '__main__':
    test_loader = BrainDataset(True, ToTensor(), ToTensor())
    model = torch.load('models/model.pth')
    device = torch.device('cuda')
    model.eval().to(device)

    for img, gt in test_loader:
        img, gt = img.to(device), gt.to(device)
        img = img.unsqueeze(0)
        pred = model(img.to(device)).squeeze(0)
        img = img.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        gt = gt.detach().cpu().numpy().transpose(1, 2, 0)
        pred = (pred > .9).float().detach().cpu().numpy()\
            .transpose(1, 2, 0) * 255
        cv.imshow('Input Image', img)
        cv.imshow('Model Prediction', pred)
        cv.imshow('Ground Truth', gt)
        cv.waitKey(250)
