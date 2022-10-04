import torch
from torch import nn
from typing import Callable
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader


def default_kd_loss(student_out, teacher_out, label, t=5):
    kl_div = F.kl_div(F.log_softmax(student_out / t, dim=1), torch.softmax(teacher_out / t, dim=1))
    if label is None:
        return kl_div
    cross_entropy = F.cross_entropy(student_out, label)
    return cross_entropy + kl_div


def default_optimizer(model: nn.Module, lr=1e-1, ) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=lr, nesterov=True, momentum=0.9)


def default_lr_scheduler(optimizer):
    class ALRS():
        '''
        proposer: Huanran Chen
        theory: landscape
        Bootstrap Generalization Ability from Loss Landscape Perspective
        '''

        def __init__(self, optimizer, loss_threshold=0.02, loss_ratio_threshold=0.02, decay_rate=0.97):
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


class NonRobustFeatureKD():
    def __init__(self, teacher: nn.Module, student: nn.Module,
                 loss_function: Callable or None = None, optimizer: torch.optim.Optimizer or None = None,
                 scheduler=None,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.teacher = teacher
        self.student = student
        self.criterion = loss_function if loss_function is not None else default_kd_loss
        self.optimizer = optimizer if optimizer is not None else default_optimizer(self.student)
        self.scheduler = scheduler if scheduler is not None else default_lr_scheduler(self.optimizer)
        self.device = device

        # initialization
        self.ban_teacher_gradient()

    def ban_teacher_gradient(self):
        for module in self.teacher.modules():
            module.requires_grad_(False)

    @staticmethod
    def clamp(x: torch.tensor, min=0, max=1):
        return torch.clamp(x, min=min, max=max)

    def generate_data(self, device, **kwargs):
        '''
        generate non-robust features
        :return: detached x, y
        '''
        # attention: x is mean 0 and std 1, please make sure teacher is desirable for this kind of input!!!
        x = torch.randn(*kwargs['size'], requires_grad=True, device=device)
        y = torch.randint(low=0, high=kwargs['max_num_classes'], size=(kwargs['size'][0]), device=device)
        for step in range(kwargs['iter_step']):
            pre = self.teacher(x)  # N, num_classes
            loss = F.cross_entropy(pre, y)
            loss.backward()
            grad = x.grad
            with torch.no_grad():
                x = x - kwargs['lr'] * grad
        return x.detach(), y.detach()

    def train(self,
              total_epoch=10,
              step_each_epoch=6000,
              fp16=True,
              generating_data_configuration={'iter_step': 3,
                                             'lr': 0.1,
                                             'max_num_classes': 1000,
                                             'size': (64, 3, 32, 32)
                                             }
              ):
        '''

        :param total_epoch:
        :param step_each_epoch: this 2 parameters is just a convention, for when output loss and acc, etc.
        :param fp16:
        :param generating_data_configuration:
        :return:
        '''
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        self.teacher.eval()
        self.student.train()
        for epoch in range(1, total_epoch + 1):
            train_loss = 0
            train_acc = 0
            for step in range(1, step_each_epoch + 1):
                x, y = self.generate_data(self.device, **generating_data_configuration)
                with torch.no_grad():
                    teacher_out = self.teacher(x)
                if fp16:
                    with autocast():
                        student_out = self.student(x)  # N, 60
                        _, pre = torch.max(student_out, dim=1)
                        loss = self.criterion(student_out, teacher_out, y)
                else:
                    x = self.student(x)  # N, 60
                    _, pre = torch.max(x, dim=1)
                    loss = self.criterion(student_out, teacher_out, y)

                if pre.shape != y.shape:
                    _, y = torch.max(y, dim=1)
                train_acc += (torch.sum(pre == y).item()) / y.shape[0]
                train_loss += loss.item()
                self.optimizer.zero_grad()

                if fp16:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_value_(self.student.parameters(), 0.1)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_value_(self.student.parameters(), 0.1)
                    self.optimizer.step()

            train_loss /= step_each_epoch
            train_acc /= step_each_epoch

            print(f'epoch {epoch}, test loader loss = {train_loss}, acc = {train_acc}')
            torch.save(self.student.state_dict(), 'student.pth')
