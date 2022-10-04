from NonRobustFeatureKD import NonRobustFeatureKD
from torchvision import models

teacher = models.resnet50(pretrained=True)
student = models.resnet18()

a = NonRobustFeatureKD(teacher, student)
a.train()