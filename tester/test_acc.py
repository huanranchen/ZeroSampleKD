import torch
from torch.utils.data import DataLoader
from torch import nn


@torch.no_grad()
def test_acc(model: nn.Module, loader: DataLoader):
    total_loss = 0
    total_acc = 0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        pre = model(x)
        total_loss += criterion(pre, y).item()
        _, pre = torch.max(pre, dim=1)
        total_acc += torch.sum((pre == y)).item() / y.shape[0]

    test_loss = total_loss / len(loader)
    test_accuracy = total_acc / len(loader)
    print(f'loss = {test_loss}, acc = {test_accuracy}')
    return test_loss, test_accuracy
