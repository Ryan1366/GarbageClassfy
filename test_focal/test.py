import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset

from tqdm import tqdm

from data_load import Date_Loader    
from model.resnet import resnet18
from losses import *
from confusionmartix import ConfusionMatrix

random_state = 1
torch.manual_seed(random_state)  
torch.cuda.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)
np.random.seed(random_state)

num_workers = 8  # 多线程的数目
use_gpu = torch.cuda.is_available()


test_dataset = Date_Loader("/home/yijiahe/you/dataset/garbage_classfily/val.txt")#####测试数据集的地址
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=4)


net = resnet18(num_classes = 4)

net.load_state_dict(torch.load('./resnet18_focal_best_299_acc_96.34869.pth'))     ## 加载resnet18 模型

if use_gpu:
    net = net.cuda()

loss = 'focal_loss'

if loss == 'focal_loss':
    criterion = FocalLoss(gamma=2)
else:
    criterion = torch.nn.CrossEntropyLoss()

if __name__ == '__main__':
    # 模型测试
    correct = 0
    test_loss = 0.0
    test_total = 0
    test_total = 0
    labels = ['clean','unclean','stains','others']
    confusion = ConfusionMatrix(num_classes=4, labels=labels)
    net.eval()
    for data in tqdm(test_loader):
        images, labels = data
        if use_gpu:
            images, labels = Variable(images.cuda()), Variable(labels.cuda())
        else:
            images, labels = Variable(images), Variable(labels)
        outputs = net(images)

        outputs_1 = torch.softmax(outputs, dim=1)
        print('outputs_1:',outputs_1)
        outputs_2 = torch.argmax(outputs_1, dim=1)
        print('outputs_2:',outputs_2)

        _, predicted = torch.max(outputs.data, 1)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        test_total += labels.size(0)
        correct += (predicted == labels.data).sum()
        confusion.update(outputs_2.cpu().detach().numpy(), labels.cpu().detach().numpy())
    print('test epoch loss: %.3f  acc: %.3f ' % ( test_loss / test_total, 100 * correct / test_total))
    confusion.plot()
    confusion.summary()

