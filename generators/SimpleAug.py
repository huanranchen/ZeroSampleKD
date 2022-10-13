import torch
from kornia import augmentation as KA
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

p = lambda x: nn.Parameter(torch.tensor(x))


def default_generator_loss(student_out, teacher_out, label, alpha=1, beta=1):
    t_loss = F.cross_entropy(teacher_out, label)
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


class CIFARBaseline(nn.Module):
    def __init__(self, student: nn.Module, teacher: nn.Module, config=default_generating_configuration()):
        super(CIFARBaseline, self).__init__()
        self.aug = KA.AugmentationSequential(
            KA.ColorJitter(p(0.1), p(0.1), p(0.1)),
            KA.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
        )
        self.student = student
        self.teacher = teacher
        self.config = config
        self.optimizer = torch.optim.SGD(self.parameters(), lr=config['lr'], momentum=0.9)

    def forward(self, x):
        x = self.aug(x)
        return x
