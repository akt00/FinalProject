import math

import onnx
import onnxsim
import torch
from torch.optim.lr_scheduler import _LRScheduler
from metrics import iou, sensitivity, specificity


# the code in this section is originally written by Akihiro Tanimoto
def train_one_epoch(model, loss_fn, optimizer, data_loader,
                    device=torch.device('cuda')):
    """ Trains the model for one epoch

    Supports only the standard backprop path

    Args:
        model: torch's model
        loss_fn: loss function
        optimizer: torch's optimizer
        data_loader: torch's DataLoader object that wraps's the Dataset object
        device (torch.device): device type (cpu|cuda)
    """
    model.train().to(device)
    batch_size = None
    dataset_size = len(data_loader.dataset)
    avg_loss = 0
    for batch, (images, targets) in enumerate(data_loader):
        if batch_size is None:
            batch_size = len(images)
        optimizer.zero_grad()
        images, targets = images.to(device), targets.to(device)
        preds = model(images)
        loss = loss_fn(preds, targets)
        avg_loss += loss
        loss.backward()
        optimizer.step()
        # prints out the current loss after each 25 batches
        if batch % 25 == 0:
            loss, current = loss.item(), (batch + 1) * batch_size
            if current > dataset_size:
                current = dataset_size
            print(f'loss: {loss:.5f} [{current}/{dataset_size}]')

    return avg_loss / batch
# the code ends here


# the code in this section is originally written by Akihiro Tanimoto
def evaluate(model, loss_fn, data_loader, device=torch.device('cuda')):
    """ Evalutes the model on the test dataset

    Supported metrics: iou, specificity, sensitivity, youden's j index

    Args:
        model (torch.nn.Module): PyTorch's model
        loss_fn (torch.nn.Module): loss function
        data_loader: torch's DataLoader for test
        device (torch.device): device type (cpu|cuda)
    """
    model.eval().to(device)
    num_batches = len(data_loader)
    test_loss, test_miou, test_sensitivity, test_specificity = .0, .0, .0, .0
    with torch.no_grad():
        for images, targets in data_loader:
            images, targets = images.to(device), targets.to(device)
            preds = model(images)
            test_loss += loss_fn(preds, targets).item()
            test_miou += iou((preds > .9).float(), targets).mean(0).item()
            test_sensitivity += sensitivity((preds > .9).float(),
                                            targets).mean(0).item()
            test_specificity += specificity((preds > .9).float(),
                                            targets).mean(0).item()

    # iou, sensitivity, specificity, youden's j index
    test_loss /= num_batches
    test_miou /= num_batches
    test_sensitivity /= num_batches
    test_specificity /= num_batches
    j_index = test_sensitivity + test_specificity - 1

    print(f'Test Metrics: \n    Loss: {test_loss:.5f} mIoU: {test_miou:.5f}'
          f' Sensitivity: {test_sensitivity:.5f} Specificity: '
          f'{test_specificity:.5f} Youden\'s J Index: {j_index:.5f}')

    return test_loss, test_miou, test_sensitivity, test_specificity, j_index
# the code ends here


# the code in this section is originally written by Akihiro Tanimoto
def export2onnx(model: torch.nn.Module, model_name: str, input_shape: tuple,
                device=torch.device('cuda'), simplify=True) -> None:
    """ Exports pytorch model to ONNX format

    model (torch.nn.Module): PyTorch model
    model_name (str): the name of the exported model with onnx extention
    input_shape (tuple): the shape of the input to the onnx model. (dynamic
        input is not yet supported)
    device (torch.device): torch's device (cpu|cuda)
    simplify (bool): if true, the model representation in onnx is simplified.
    """
    x = torch.randn(*input_shape, requires_grad=True).to(device=device)
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
                 last_epoch: int = -1, max_epoch: int = 30):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multipiler should be 1 or above')
        self.warmup_epoch = warmup_epoch
        self.last_epoch = last_epoch
        self.eta_min = min_lr
        self.T_max = float(epochs - warmup_epoch)
        self.after_scheduler = True
        self.max_epoch = max_epoch
        super(WarmupCosineAnnealingLR, self).__init__(optimizer=optimzer,
                                                      last_epoch=last_epoch)

    def get_lr(self):
        if self.max_epoch < self.last_epoch:
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


model = torch.nn.Sequential(
    torch.nn.Linear(32, 32)
)
opt = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=.9, nesterov=True)
scheduler = WarmupCosineAnnealingLR(
    opt, 1, 4, 30
)
lrs = []
for ep in range(50):
    lrs.append(*scheduler.get_lr())
    opt.step()
    scheduler.step()


print(scheduler.last_epoch, scheduler.get_lr())
import matplotlib.pyplot as plt
plt.plot(lrs)
plt.show()
