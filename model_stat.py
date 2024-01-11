
# ----- 打印模型的计算量和参数数量
import torch 
import torch.nn as nn

from torchstat import stat
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, resnext101_32x8d, mobilenet_v3_small, mobilenet_v3_large
 

'''
#net = resnet18(pretrained=True)
net = resnet18()
net.fc = torch.nn.Linear(512, 9)
net.eval()
stat(net, (3, 112, 112))
total = sum([param.nelement() for param in net.parameters()])
print("Number of parameters: %.2fM" % (total/1e6))
'''

###stat计算
net = mobilenet_v3_large()
num_ftrs = net.classifier[3].in_features
net.classifier[3] = nn.Linear(num_ftrs, 9)
net.eval()
stat(net, (3, 112, 112))
total = sum([param.nelement() for param in net.parameters()])
print("Number of parameters: %.2fM" % (total/1e6))
