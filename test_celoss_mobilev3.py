import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset

from tqdm import tqdm
from data_load import Date_Loader    
from torchvision.models import *
from labelsmoothing import CELoss


# 配置参数
random_state = 1
torch.manual_seed(random_state)  # 设置随机数种子，确保结果可重复
torch.cuda.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)
np.random.seed(random_state)


use_gpu = torch.cuda.is_available()


# train_dataset = Date_Loader(path="/home/yijiahe/you/dataset/cls9/train.txt") 
# print("train_loader:",len(train_dataset))
# train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle = True, num_workers=4, drop_last = True)


test_dataset = Date_Loader(path="/home/yijiahe/you/dataset/cls9/val.txt",train_flag = False)
print("test_dataset:",len(test_dataset))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle = True)


net = mobilenet_v3_large(pretrained=True)
num_ftrs = net.classifier[3].in_features
net.classifier[3] = nn.Linear(num_ftrs, 9)


pretrain = 1
if pretrain:
    print("load pretrain model!")
    net.load_state_dict(torch.load('/home/yijiahe/you/garbage_c4/weight_c9/train_celoss_mobilev3/resnet18_best_70_acc_98.15522.pth'))

if use_gpu:
    net = net.cuda()


cirterion = CELoss(label_smooth=0.05, class_num=9)   


if __name__ == '__main__':

    net.eval()
    correct = 0
    test_loss = 0.0
    test_total = 0
    with torch.no_grad():
        for data in tqdm(test_loader):
            inputs, labels, img_path = data
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            outputs = net(inputs)
            _, predict = torch.max(outputs.data, 1)
            loss = cirterion(outputs, labels)
            test_loss += loss
            test_total += labels.size(0)
            correct += (predict == labels.data).sum() 
            print("labels, predict:",labels, predict)
        print('test epoch loss: %.3f  acc: %.3f ' % ( test_loss.sum() / test_total, 100 * correct / test_total)) 


