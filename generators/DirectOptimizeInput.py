import torch
from torch import nn
import torch.nn.functional as F


def default_generator_loss(student_out, teacher_out=None, label=None, alpha=1, beta=1):
    if teacher_out:
        t_loss = F.cross_entropy(teacher_out, label)
    else:
        t_loss = 0
    s_loss = F.cross_entropy(student_out, label)
    return alpha * t_loss - beta * s_loss


def default_generating_configuration():
    x = {'iter_step': 1,
         'lr': 1e-4,
         'criterion': default_generator_loss,
         }
    print('generating config:')
    print(x)
    print('-' * 100)
    return x


class DirectOptimizeInput():
    def __init__(self, student: nn.Module, teacher: nn.Module = None,
                 config=default_generating_configuration()):
        self.student = student
        self.teacher = teacher
        self.config = config

    def generate_data(self, x, y, ):
        '''
        generate input
        :return: detached x, y
        '''
        self.student.requires_grad_(False)
        self.student.eval()
        # attention: x is mean 0 and std 1, please make sure teacher is desirable for this kind of input!!!
        original_x = x.clone()
        original_y = y.clone()
        x.requires_grad = True
        for step in range(self.config['iter_step']):
            if self.teacher:
                loss = self.config['criterion'](self.student(x), self.teacher(x), y)
            else:
                loss = self.config['criterion'](self.student(x), y)
            loss.backward()
            grad = x.grad
            x.requires_grad = False
            x = x - self.config['lr'] * grad.sign()
            # print(torch.mean(torch.abs(x)).item())
            # print(torch.abs(x))
            # x = x - kwargs['lr'] * grad
            x.requires_grad = True

        self.student.requires_grad_(True)
        self.student.train()

        return torch.cat([x.detach(), original_x], dim=0), \
               torch.cat([y.detach(), original_y], dim=0)

    def __call__(self, *args, **kwargs):
        return self.generate_data(*args, **kwargs)
