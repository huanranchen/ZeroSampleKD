import torch
from torch import nn
from typing import Callable
from torch.nn import functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tester import test_acc
from optimizer import default_optimizer, default_lr_scheduler


def default_kd_loss(student_out, teacher_out=None, label=None, t=5, alpha=1, beta=1):
    kl_div = F.kl_div(F.log_softmax(student_out / t, dim=1), torch.softmax(teacher_out / t, dim=1),
                      reduction='batchmean')
    if label is None:
        return kl_div
    cross_entropy = F.cross_entropy(student_out, label)
    return alpha * cross_entropy + beta * kl_div


def default_generator_loss(student_out, teacher_out, label, alpha=1, beta=1):
    t_loss = F.cross_entropy(teacher_out, label)
    s_loss = F.cross_entropy(student_out, label)
    return alpha * t_loss - beta * s_loss


def default_generating_configuration():
    x = {'iter_step': 1,
         'lr': 1e-4,
         'criterion': default_generator_loss,
         }
    return x


class LearnWhatYouDontKnow():
    def __init__(self, teacher: nn.Module, student: nn.Module,
                 loss_function: Callable or None = None, optimizer: torch.optim.Optimizer or None = None,
                 scheduler=None,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 eval_loader: DataLoader = None):
        self.teacher = teacher
        self.student = student
        self.criterion = loss_function if loss_function is not None else default_kd_loss
        self.optimizer = optimizer if optimizer is not None else default_optimizer(self.student)
        self.scheduler = scheduler if scheduler is not None else default_lr_scheduler(self.optimizer)
        self.device = device
        self.eval_loader = eval_loader

        # initialization
        self.init()

    def init(self):
        # ban teacher gradient
        self.teacher.requires_grad_(False)

        # change device
        self.teacher.to(self.device)
        self.student.to(self.device)

        # tensorboard
        self.writer = SummaryWriter(log_dir="runs/1e-4")

    def generate_data(self, x, y, **kwargs):
        '''
        generate input
        :return: detached x, y
        '''
        self.student.requires_grad_(False)
        # attention: x is mean 0 and std 1, please make sure teacher is desirable for this kind of input!!!
        original_x = x.clone()
        original_y = y.clone()
        x.requires_grad = True
        for step in range(kwargs['iter_step']):
            loss = kwargs['criterion'](self.student(x), self.teacher(x), y)
            loss.backward()
            grad = x.grad
            x.requires_grad = False
            # x = x - kwargs['lr'] * grad.sign()
            # print(torch.mean(torch.abs(x)).item())
            # print(torch.abs(x))
            x = x - kwargs['lr'] * grad
            x.requires_grad = True

        self.student.requires_grad_(True)

        return torch.cat([x.detach(), original_x], dim=0), \
               torch.cat([y.detach(), original_y], dim=0)

    def train(self,
              loader: DataLoader,
              total_epoch=120,
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
            student_confidence = 0
            teacher_confidence = 0
            pbar = tqdm(loader)
            for step, (x, y) in enumerate(pbar, 1):
                x, y = x.to(self.device), y.to(self.device)
                x, y = self.generate_data(x, y, **generating_data_configuration)
                with torch.no_grad():
                    teacher_out = self.teacher(x)
                    teacher_confidence += torch.mean(
                        F.softmax(teacher_out, dim=1)[torch.arange(y.shape[0] // 2), y[:y.shape[0] // 2]]).item()

                if fp16:
                    with autocast():
                        student_out = self.student(x)  # N, 60
                        _, pre = torch.max(student_out, dim=1)
                        loss = self.criterion(student_out, teacher_out, y)
                else:
                    student_out = self.student(x)  # N, 60
                    _, pre = torch.max(student_out, dim=1)
                    loss = self.criterion(student_out, teacher_out, y)

                student_confidence += torch.mean(
                    F.softmax(student_out, dim=1)[torch.arange(y.shape[0] // 2), y[:y.shape[0] // 2]]).item()

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

            print(f'epoch {epoch}, loss = {train_loss}, acc = {train_acc}')
            torch.save(self.student.state_dict(), 'student.pth')

            # tensorboard
            self.writer.add_scalar('confidence/teacher_confidence', teacher_confidence / len(loader), epoch)
            self.writer.add_scalar('confidence/student_confidence', student_confidence / len(loader), epoch)

            self.writer.add_scalar('train/loss', train_loss, epoch)
            self.writer.add_scalar('train/acc', train_acc, epoch)
            if self.eval_loader is not None:
                test_loss, test_accuracy = test_acc(self.student, self.eval_loader, self.device)
                self.writer.add_scalar('test/loss', test_loss, epoch)
                self.writer.add_scalar('test/acc', test_accuracy, epoch)


if __name__ == '__main__':
    from backbones import wrn_40_2, wrn_16_2
    from data import get_CIFAR100_train, get_CIFAR100_test

    teacher = wrn_40_2(num_classes=100)
    teacher.load_state_dict(torch.load('./checkpoints/wrn_40_2.pth')['model'])
    student = wrn_16_2(num_classes=100)

    loader: DataLoader = get_CIFAR100_train()

    solver = LearnWhatYouDontKnow(teacher, student, eval_loader=get_CIFAR100_test())
    solver.train(loader)
