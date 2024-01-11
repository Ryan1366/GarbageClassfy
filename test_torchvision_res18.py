import torch
import numpy as np
import torch.nn as nn
import torchvision.models as models

from torch.autograd import Variable
from torch.utils.data import Dataset
from tqdm import tqdm
from data_load import Date_Loader    

from labelsmoothing import CELoss
from BiTemperedLogisticLoss import Bi_Tempered_Logistic_Loss

# 配置参数
random_state = 1
torch.manual_seed(random_state)  # 设置随机数种子，确保结果可重复
torch.cuda.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)
np.random.seed(random_state)
# random.seed(random_state)

use_gpu = torch.cuda.is_available()

pretrain = 1

test_dataset = Date_Loader(path="/home/yijiahe/you/dataset/cls9/val.txt",train_flag = False)
print("test_dataset:",len(test_dataset))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle = True)

net = models.resnet18()
net.fc = torch.nn.Linear(512, 9)

if pretrain:
    print("load pretrain model!")
    net.load_state_dict(torch.load('/home/yijiahe/you/garbage_c4/weight_c9/resnet18_btl_60.pth'))

if use_gpu:
    net = net.cuda()

# cirterion = nn.CrossEntropyLoss()
# cirterion = CELoss(label_smooth=0.05, class_num=9)   
cirterion = Bi_Tempered_Logistic_Loss(label_smoothing = 0.05)

if __name__ == '__main__':
    net.eval()
    correct = 0
    test_loss = 0.0
    test_total = 0
    for data in tqdm(test_loader):
        inputs, labels, img_path, bitemperedlosslabel= data
        if use_gpu:
            inputs, labels, bitemperedlosslabel= Variable(inputs.cuda()), Variable(labels.cuda()), Variable(bitemperedlosslabel.cuda())
        else:
            inputs, labels, bitemperedlosslabel = Variable(inputs), Variable(labels), Variable(bitemperedlosslabel)
        outputs = net(inputs)
        _, predict = torch.max(outputs.data, 1)
        print("labels:",labels)
        print("img_path:",img_path)
        print("predict:",predict)
        loss = cirterion(outputs, bitemperedlosslabel)
        test_loss += loss
        test_total += labels.size(0)
        correct += (predict == labels.data).sum() 
    print('test epoch loss: %.3f  acc: %.3f ' % ( test_loss.sum() / test_total, 100 * correct / test_total)) 


'''
# ----- 打印模型的计算量和参数数量
from torchstat import stat
from torchvision.models import resnet50, resnet101, resnet152, resnext101_32x8d
 
net.eval()
stat(net, (3, 32, 32))

total = sum([param.nelement() for param in net.parameters()])
print("Number of parameters: %.2fM" % (total/1e6))
'''
