from Bugs import NonRobustFeatureKD
from torchvision import models
from tester.test_acc import test_acc
from data import get_CIFAR100_test

teacher = models.convnext_tiny(pretrained=True)
student = models.convnext_tiny()

test_acc(teacher, get_CIFAR100_test())

a = NonRobustFeatureKD(teacher, student)
a.train()
