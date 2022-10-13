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


class SimpleAug(nn.Module):
    def __init__(self, student: nn.Module, teacher: nn.Module, config=default_generating_configuration(),
                 mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761]):
        super(SimpleAug, self).__init__()
        self.aug = KA.AugmentationSequential(
            # KA.Normalize([], []),  # Normalize Back
            KA.ColorJitter(p(0.1), p(0.1), p(0.1), p(0.1)),
            KA.Normalize(mean, std),
        )
        self.student = student
        self.teacher = teacher
        self.config = config
        self.optimizer = torch.optim.SGD(self.parameters(), lr=config['lr'], momentum=0.9)
        self.mean = mean
        self.std = std

    def normalize_back(self, x: torch.tensor, ):
        '''

        :param x: N, C, H, D
        :return:
        '''
        # print(x.shape, torch.tensor(self.std).shape, torch.tensor(self.mean).shape)
        x = x * torch.tensor(self.std, device=x.device).reshape(1, -1, 1, 1) \
            + torch.tensor(self.mean, device=x.device).reshape(1, -1, 1, 1)
        return x

    def forward(self, x, y):
        self.student.eval()
        self.student.requires_grad_(False)
        self.requires_grad_(True)
        original_x = x.clone()
        original_y = y.clone()

        x = self.normalize_back(x)
        x = self.aug(x)
        loss = self.config['criterion'](self.student(x), self.teacher(x), y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # give back
        self.student.train()
        self.student.requires_grad_(True)
        self.requires_grad_(False)

        # prepare for final
        x = self.normalize_back(original_x.clone())
        x = self.aug(x).detach()

        return torch.cat([x.detach(), original_x], dim=0), \
               torch.cat([y.detach(), original_y], dim=0)
