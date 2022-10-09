import torch
from torch import nn
from typing import Callable
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def default_kd_loss(student_out, teacher_out=None, label=None, t=5):
    kl_div = F.kl_div(F.log_softmax(student_out / t, dim=1), torch.softmax(teacher_out / t, dim=1),
                      reduction='batchmean')
    if label is None:
        return kl_div
    cross_entropy = F.cross_entropy(student_out, label)
    return cross_entropy + kl_div


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


def default_generator_loss(student_out, teacher_out, label, alpha=1, beta=1):
    t_loss = F.cross_entropy(teacher_out, label)
    s_loss = F.cross_entropy(student_out, label)
    return alpha * t_loss - beta * s_loss


def default_generating_configuration():
    x = {'iter_step': 5,
         'lr': 5e-3,
         'size': (256, 3, 32, 32),
         'criterion': default_generator_loss,
         }
    return x


class LearnWhatYouDontKnow():
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
        self.init()

    def init(self):
        # ban teacher gradient
        for module in self.teacher.modules():
            module.requires_grad_(False)

        # change device
        self.teacher.to(self.device)
        self.student.to(self.device)

        # tensorboard
        self.writer = SummaryWriter(log_dir="runs/result_1", flush_secs=120)

    def generate_data(self, x, y, **kwargs):
        '''
        generate input
        :return: detached x, y
        '''
        # attention: x is mean 0 and std 1, please make sure teacher is desirable for this kind of input!!!
        original_x = x.clone()
        original_y = y.clone()
        x.requires_grad = True
        for step in range(kwargs['iter_step']):
            loss = kwargs['criterion'](self.student(x), self.teacher(x), y)
            loss.backward()
            grad = x.grad
            x.requires_grad = False
            x = x - kwargs['lr'] * grad.sign()
            x.requires_grad = True

        return torch.cat([x.detach(), original_x], dim=0), \
               torch.cat([y.detach(), original_y], dim=0)

    def train(self,
              loader: DataLoader,
              total_epoch=1000,
              fp16=False,
              generating_data_configuration=default_generating_configuration()
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
            pbar = tqdm(loader)
            for step, (x, y) in enumerate(pbar, 1):
                x, y = x.to(self.device), y.to(self.device)
                x, y = self.generate_data(x, y, **generating_data_configuration)
                with torch.no_grad():
                    teacher_out = self.teacher(x)
                    self.writer.add_scalar(tag='teacher_confidence',
                                           scalar_value=torch.mean(
                                               F.softmax(teacher_out, dim=1)[torch.arange(y.shape[0]), y]).item(),
                                           global_step=len(loader) * (epoch - 1) + step)
                if fp16:
                    with autocast():
                        student_out = self.student(x)  # N, 60
                        _, pre = torch.max(student_out, dim=1)
                        loss = self.criterion(student_out, teacher_out, y)
                else:
                    student_out = self.student(x)  # N, 60
                    _, pre = torch.max(student_out, dim=1)
                    loss = self.criterion(student_out, teacher_out, y)

                self.writer.add_scalar(tag='student_confidence',
                                       scalar_value=torch.mean(
                                           F.softmax(student_out, dim=1)[torch.arange(y.shape[0]), y]).item(),
                                       global_step=len(loader) * (epoch - 1) + step)

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

                if step % 10 == 0:
                    pbar.set_postfix_str(f'loss={train_loss / step}, acc={train_acc / step}')

            train_loss /= len(loader)
            train_acc /= len(loader)

            self.scheduler.step(train_loss)

            print(f'epoch {epoch}, test loader loss = {train_loss}, acc = {train_acc}')
            torch.save(self.student.state_dict(), 'student.pth')


if __name__ == '__main__':
    from backbones import wrn_40_2, wrn_16_2
    from data import get_CIFAR100_train

    teacher = wrn_40_2(num_classes=100)
    teacher.load_state_dict(torch.load('./checkpoints/wrn_40_2.pth')['model'])
    student = wrn_16_2(num_classes=100)

    loader: DataLoader = get_CIFAR100_train()

    solver = LearnWhatYouDontKnow(teacher, student)
    solver.train(loader)
