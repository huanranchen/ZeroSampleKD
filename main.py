import torch
from backbones import wrn_40_2, wrn_16_2
from tester.test_acc import test_acc
from data import get_CIFAR100_test

student = wrn_16_2(num_classes=100)
student.load_state_dict(torch.load('student.pth'))
test_acc(student, get_CIFAR100_test())
