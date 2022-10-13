import torch
from backbones import wrn_40_2, wrn_16_2
from tester.test_acc import test_acc
from data import get_CIFAR100_test

# student = wrn_16_2(num_classes=100)
# student.load_state_dict(torch.load('student.pth'))
student = wrn_40_2(num_classes=100).cuda()
student.eval()
student.load_state_dict(torch.load('./checkpoints/wrn_40_2.pth')['model'])
test_acc(student, get_CIFAR100_test())
