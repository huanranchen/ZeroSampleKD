import torch
from torch import nn
import torch.nn.functional as F
from .DirectOptimizeInput import DirectOptimizeInput


def untarget_attack_loss(student_out, label, beta=1):
    s_loss = F.cross_entropy(student_out, label)
    return - beta * s_loss


class UntargetPGD(DirectOptimizeInput):
    def __init__(self, model: nn.Module, step_size=1e-3, iter_step=1, criterion=untarget_attack_loss):
        config = {
            'iter_step': iter_step,
            'lr': step_size,
            'criterion': criterion,
        }
        super(UntargetPGD, self).__init__(student=model, config=config)


class UntargetFGSM(DirectOptimizeInput):
    def __init__(self, model: nn.Module, step_size=1e-3, criterion=untarget_attack_loss):
        config = {
            'iter_step': 1,
            'lr': step_size,
            'criterion': criterion,
        }
        super(UntargetFGSM, self).__init__(student=model, config=config)
