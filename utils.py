import onnx
import onnxsim
import torch
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
