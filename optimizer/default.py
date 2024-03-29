import torch
from torch import nn

def default_optimizer(model: nn.Module, lr=1e-1, ) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=lr, nesterov=True, momentum=0.9)
    # return torch.optim.Adam(model.parameters(), lr=lr, )


def default_lr_scheduler(optimizer):
    class ALRS():
        '''
        proposer: Huanran Chen
        theory: landscape
        Bootstrap Generalization Ability from Loss Landscape Perspective
        '''

        def __init__(self, optimizer, loss_threshold=0.02, loss_ratio_threshold=0.02, decay_rate=0.9):
            self.optimizer = optimizer
            self.loss_threshold = loss_threshold
            self.decay_rate = decay_rate
            self.loss_ratio_threshold = loss_ratio_threshold

            self.last_loss = 999

        def step(self, loss):
            delta = self.last_loss - loss
            if delta < self.loss_threshold and delta / self.last_loss < self.loss_ratio_threshold:
                for group in self.optimizer.param_groups:
                    group['lr'] *= self.decay_rate
                    now_lr = group['lr']
                    print(f'now lr = {now_lr}')

            self.last_loss = loss

    return ALRS(optimizer)