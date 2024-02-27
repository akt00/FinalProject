import sys
import time
# external libs
import cv2 as cv
import torch
# local modules
from src.data_pipeline import BrainDatasetv2, data_pipeline
from src.utils import export2onnx


def profile():
    from torchinfo import summary
    models = [
        'models/unet.pth',
        'models/attention_gate_unet.pth',
        'models/deformable_unet.pth',
        'models/deeplabv4.pth',
    ]
    test_loader = BrainDatasetv2(False, data_pipeline(False), False)
    device = torch.device('cpu')
    input_shape = (8, 3, 256, 256)
    for m in models:
        model = torch.load(m)
        summary(model, input_size=input_shape)
        model.eval().to(device)
        for img, gt in test_loader:
            img, gt = img.to(device), gt.to(device)
            start = time.time()
            img = img.unsqueeze(0)
            pred = model(img).squeeze(0)
            pred = (pred > .9).float().detach().cpu().numpy()\
                .transpose(1, 2, 0) * 255
            print(f'Inference Time: {time.time() - start:.4f}')
            break


def export():
    from src.models import (SPPModule, ResidualBlock,
                            SEModule, UNet)
    device = torch.device('cuda')
    spp = SPPModule(64)
    export2onnx(spp.to(device), 'spp_module.onnx', (8, 64, 64, 64))
    res = ResidualBlock(64, True, True)
    export2onnx(res.to(device), 'residual_block.onnx', (8, 64, 64, 64))
    se = SEModule(64)
    export2onnx(se.to(device), 'squeeze_and_excitation.onnx', (8, 64, 64, 64))
    unet = UNet(3, 1, attention_gate=False, dcn=False)
    export2onnx(unet.to(device), 'unet.onnx', (8, 3, 256, 256))


def video_demo():
    """ shows model predictions on test dataset """
    test_loader = BrainDatasetv2(False, data_pipeline(False), False)
    model = torch.load('models/deeplabv4.pth')
    device = torch.device('cuda')
    model.eval().to(device)

    for img, gt in test_loader:
        img, gt = img.to(device), gt.to(device)
        img = img.unsqueeze(0)
        pred = model(img).squeeze(0)
        img = img.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0)
        gt = gt.detach().cpu().numpy().transpose(1, 2, 0)
        pred = (pred > .9).float().detach().cpu().numpy()\
            .transpose(1, 2, 0) * 255
        cv.imshow('Input Image', img)
        cv.imshow('Model Prediction', pred)
        cv.imshow('Ground Truth', gt)
        cv.waitKey(200)


if __name__ == '__main__':
    args = sys.argv
    assert len(args) == 2, 'It should be either one of [eval, export, profile]'
    mode = args[-1]
    if mode == 'eval':
        video_demo()
    elif mode == 'export':
        export()
    elif mode == 'profile':
        profile()
    else:
        raise Exception('It should be either one of [eval, export, profile]')
