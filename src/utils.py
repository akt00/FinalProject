import math
# external libs
import onnx
import onnxsim
import torch
from torch import device, Tensor
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# local modules
from .metrics import iou, precision, sensitivity, f1_score
from .data_pipeline import data_pipeline, BrainDatasetv2
from .losses import FocalDiceLoss


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
    avg_loss = 0
    for batch, (images, targets) in tqdm(enumerate(data_loader)):
        if batch_size is None:
            batch_size = len(images)
        optimizer.zero_grad()
        images, targets = images.to(device=dev), targets.to(device=dev)
        preds = model(images)
        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
    print(f'[Train]  Loss: {avg_loss / batch:.4f}')
    return avg_loss / batch


def evaluate(model: Module, loss_fn: Module, data_loader: DataLoader,
             dev: device = device('cuda')) -> tuple:
    """ Evalutes the model on the test dataset

    Args:
        model: PyTorch's model
        loss_fn: loss function
        data_loader: torch's DataLoader for test
        device: device type (cpu|cuda)
    """
    model.eval().to(device=dev)
    num_batches = len(data_loader)
    _loss, _miou = .0, .0
    _precision, _recall, _f1 = .0, .0, .0
    with torch.no_grad():
        for images, targets in data_loader:
            images, targets = images.to(device=dev), targets.to(device=dev)
            preds = model(images)
            _loss += loss_fn(preds, targets).item()
            _miou += iou((preds > .5).float(), targets).mean(0).item()
            _precision += precision((preds > .5).float(), targets
                                    ).mean(0).item()
            _recall += sensitivity((preds > .5).float(), targets
                                   ).mean(0).item()
            _f1 += f1_score((preds > .5).float(), targets).mean(0).item()
    # loss, miou, precision, recall, f1 score
    _loss /= num_batches
    _miou /= num_batches
    _precision /= num_batches
    _recall /= num_batches
    _f1 /= num_batches
    # shows the current metrics
    print(f'[Test]  Loss: {_loss:.4f} mIoU: {_miou:.4f}'
          f' Precision: {_precision:.4f} Recall: {_recall:.4f} F1: {_f1:.4f}')
    return _loss, _miou, _precision, _recall, _f1


def train_and_evaluate(model, augment: bool, oversample: bool) -> None:
    train_pipeline = data_pipeline(True) if augment else data_pipeline(False)
    train_loader = BrainDatasetv2(True, train_pipeline, oversample)
    train_loader = DataLoader(train_loader, batch_size=64, shuffle=True,
                              num_workers=6, persistent_workers=True)
    test_pipeline = data_pipeline(False)
    test_loader = BrainDatasetv2(False, test_pipeline, oversample)
    test_loader = DataLoader(test_loader, batch_size=64, shuffle=False,
                             num_workers=4, persistent_workers=True)
    loss = FocalDiceLoss()
    optimizer = torch.optim.RAdam(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    min_test_loss = 1e10  # keeps the minimum validation loss
    writer = SummaryWriter()  # logs evaluation metrics
    # train and evaluate
    for epoch in range(30):
        print(f'--------- Epoch {epoch + 1} ---------')
        print(f'Current learning rate: {scheduler.get_last_lr()[0]:.5f}')
        # train for one epoch
        train_loss = train_one_epoch(model, loss, optimizer, train_loader)
        writer.add_scalar('Loss/Train', train_loss, epoch)
        # evaluate the model
        (test_loss, test_miou, test_precision, test_sensitivity, test_f1) = \
            evaluate(model, loss, test_loader)
        writer.add_scalar('Loss/Test', test_loss, epoch)
        writer.add_scalar('mIoU/Test', test_miou, epoch)
        writer.add_scalar('Precision/Test', test_precision, epoch)
        writer.add_scalar('Sensitivity/Test', test_sensitivity, epoch)
        writer.add_scalar("F1 Score/Test", test_f1, epoch)
        # save the best model
        if test_loss < min_test_loss:
            min_test_loss = test_loss
            torch.save(model, 'models/model.pth')
        scheduler.step()


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


class WarmupCosineAnnealingLR(_LRScheduler):
    """ Cosine Annealing learning rate scheduler with warmup """
    def __init__(self, optimzer: torch.optim.Optimizer, warmup_epoch: int,
                 epochs: int, min_lr: float = 1e-5, last_epoch: int = -1
                 ) -> None:
        self.warmup_epoch = warmup_epoch
        self.last_epoch = last_epoch
        self.eta_min = min_lr
        self.T_max = float(epochs - warmup_epoch)
        self.after_scheduler = True
        super().__init__(optimizer=optimzer, last_epoch=last_epoch)

    def get_lr(self) -> list:
        """ returns the learning rate """
        # returns the minimum lr after the max iterations
        if self.T_max + self.warmup_epoch < self.last_epoch:
            return [self.eta_min for _ in self.base_lrs]
        # cosine annealing lr
        elif self.last_epoch > self.warmup_epoch - 1:
            return [self.eta_min + (base_lr - self.eta_min) *
                    (1 + math.cos(math.pi * (self.last_epoch -
                     self.warmup_epoch) / (self.T_max - 1))) / 2
                    for base_lr in self.base_lrs]
        # warmup
        else:
            return [base_lr * (float(self.last_epoch + 1) / self.warmup_epoch)
                    for base_lr in self.base_lrs]
