import math
# 3rd party libs
import onnx
import onnxsim
import torch
from torch import device, Tensor
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader
# local modules
from .metrics import iou, mean_ap, precision, sensitivity, f1_score


# the code in this section is originally written by Akihiro Tanimoto
def train_one_epoch(model: Module, loss_fn: Module, optimizer: Optimizer,
                    data_loader: DataLoader, dev: device = device('cuda')
                    ) -> Tensor:
    """ Trains the model for one epoch

    Args:
        model: torch's model
        loss_fn: loss function
        optimizer: torch's optimizer
        data_loader: torch's DataLoader
        device: device type (cpu|cuda)
    """
    model.train().to(device=dev)
    batch_size = None
    dataset_size = len(data_loader.dataset)
    avg_loss = 0
    for batch, (images, targets) in enumerate(data_loader):
        if batch_size is None:
            batch_size = len(images)
        optimizer.zero_grad()
        images, targets = images.to(device=dev), targets.to(device=dev)
        preds = model(images)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        # prints out the current loss after each 25 batches
        if batch % 25 == 0:
            loss, current = loss.item(), (batch + 1) * batch_size
            if current > dataset_size:
                current = dataset_size
            print(f'loss: {loss:.5f} [{current}/{dataset_size}]')

    return avg_loss / batch
# the code ends here


# the code in this section is originally written by Akihiro Tanimoto
def evaluate(model: Module, loss_fn: Module, data_loader: DataLoader,
             dev: device = device('cuda')) -> tuple:
    """ Evalutes the model on the test dataset

    Supported metrics: iou, specificity, sensitivity, youden's j index

    Args:
        model: PyTorch's model
        loss_fn: loss function
        data_loader: torch's DataLoader for test
        device: device type (cpu|cuda)
    """
    model.eval().to(device=dev)
    num_batches = len(data_loader)
    test_loss, test_miou, test_map = .0, .0, .0, .0
    test_precision, test_sensitivity, test_f1 = .0, .0, .0
    with torch.no_grad():
        for images, targets in data_loader:
            images, targets = images.to(device=dev), targets.to(device=dev)
            preds = model(images)
            test_loss += loss_fn(preds, targets).item()
            test_miou += iou((preds > .5).float(), targets).mean(0).item()
            test_map += mean_ap((preds > .5).float(), targets).mean(0).item()
            test_precision += precision((preds > .5).float(), targets).mean(0).item()
            test_sensitivity += sensitivity((preds > .5).float(), targets).mean(0).item()
            test_f1 += f1_score((preds > .5).float(), targets).mean(0).item()
    # iou, sensitivity, specificity, youden's j index
    test_loss /= num_batches
    test_miou /= num_batches
    test_map /= num_batches
    test_precision /= num_batches
    test_sensitivity /= num_batches
    test_f1 /= num_batches

    print(f'Test Metrics: \n    Loss: {test_loss:.4f} mIoU: {test_miou:.4f}'
          f' mAP: {test_map:.4f} Precision: {test_precision:.4f}'
          f' Recall: {test_sensitivity:.4f} F1: {test_f1:.4f}')

    return test_loss, test_miou, test_map, test_precision, test_sensitivity, test_f1
# the code ends here


# the code in this section is originally written by Akihiro Tanimoto
def export2onnx(model: torch.nn.Module, model_name: str, input_shape: tuple,
                dev: device = device('cuda'), simplify=True) -> None:
    """ Exports pytorch model to ONNX format

    model: PyTorch model
    model_name: the name of the exported model with onnx extention
    input_shape: the shape of the input to the onnx model (b, ch, h, w)
    device: torch's device (cpu|cuda)
    simplify: if true, the model representation in onnx is simplified.
    """
    x = torch.randn(*input_shape, requires_grad=True).to(device=dev)
    torch.onnx.export(
        model,
        x,
        model_name,
        opset_version=17
    )
    # simplifies the model representation
    if simplify:
        model = onnx.load(model_name)
        model_sim, check = onnxsim.simplify(model)
        assert check
        onnx.save(model_sim, model_name)
# the code ends here


class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimzer: torch.optim.Optimizer, multiplier: float,
                 warmup_epoch: int, epochs: int, min_lr: float = 1e-5,
                 last_epoch: int = -1) -> None:
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multipiler should be 1 or above')
        self.warmup_epoch = warmup_epoch
        self.last_epoch = last_epoch
        self.eta_min = min_lr
        self.T_max = float(epochs - warmup_epoch)
        self.after_scheduler = True
        super(WarmupCosineAnnealingLR, self).__init__(optimizer=optimzer,
                                                      last_epoch=last_epoch)

    def get_lr(self):
        if self.T_max + self.warmup_epoch < self.last_epoch:
            return [self.eta_min for _ in self.base_lrs]
        elif self.last_epoch > self.warmup_epoch - 1:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * (self.last_epoch -
                     self.warmup_epoch) / (self.T_max - 1))) / 2
                    for base_lr in self.base_lrs]
        # warmup
        if self.multiplier == 1.:
            return [base_lr * (float(self.last_epoch + 1) / self.warmup_epoch)
                    for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch /
                    self.warmup_epoch + 1.)
                    for base_lr in self.base_lrs]
