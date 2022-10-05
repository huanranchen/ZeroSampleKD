import torch

from Bugs import NonRobustFeatureKD
from torchvision import models
from tester.test_acc import test_acc
from data import get_CIFAR10_test

teacher = models.resnet50(num_classes=10)
teacher.load_state_dict(torch.load('resnet50.pth'))
student = models.resnet50()

# test_acc(teacher, get_CIFAR10_test())

a = NonRobustFeatureKD(teacher, student)
a.train()
