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

epochs = 200  
batch_size = 8
num_workers = 8  # 多线程的数目
use_gpu = torch.cuda.is_available()

pretrain = 0

train_dataset = Date_Loader(path="/home/yijiahe/you/dataset/cls9/train.txt")  #
print("train_loader:",len(train_dataset))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle = True, num_workers=4, drop_last = True)

test_dataset = Date_Loader(path="/home/yijiahe/you/dataset/cls9/val.txt",train_flag = False)
print("test_dataset:",len(test_dataset))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle = True)

net = models.resnet18(pretrained=True)
net.fc = torch.nn.Linear(512, 9)

if pretrain:
    print("load pretrain model!")
    net.load_state_dict(torch.load('/home/yijiahe/you/garbage_c4/weight_c9/resnet18_160.pth'))

if use_gpu:
    net = net.cuda()
print(net)

# cirterion = nn.CrossEntropyLoss()
# cirterion = CELoss(label_smooth=0.05, class_num=9)   
cirterion = Bi_Tempered_Logistic_Loss(label_smoothing = 0.05)
optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer, T_max =  epochs) #  * iters 

if __name__ == '__main__':
    for epoch in tqdm(range(epochs)):

        net.train()
        total_loss = 0.0
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        acc = 0
        for i, data in enumerate(train_loader):
            inputs, train_labels,img_path, bitemperedlosslabel = data
            if use_gpu:
                inputs, labels, bitemperedlosslabel= Variable(inputs.cuda()), Variable(train_labels.cuda()), Variable(bitemperedlosslabel.cuda())
            else:
                inputs, labels, bitemperedlosslabel = Variable(inputs), Variable(train_labels), Variable(train_labels), Variable(bitemperedlosslabel)
            optimizer.zero_grad()
            outputs = net(inputs)
            _, train_predicted = torch.max(outputs.data, 1)
            train_correct += (train_predicted == labels.data).sum()
            loss = cirterion(outputs, bitemperedlosslabel)
            loss.sum().backward()
            optimizer.step()
            total_loss += loss
            #print("total_loss:",total_loss)
            train_total += train_labels.size(0)
        scheduler.step()
        print("Learning_rate:",optimizer.state_dict()['param_groups'][0]['lr'])
        cur_acc = 100 * train_correct / train_total
        print('train %d epoch loss: %.3f  acc: %.3f ' % (epoch + 1, total_loss.sum() / train_total * batch_size, cur_acc))

        if epoch % 10 == 0 and epoch > 20:
            torch.save(net.state_dict(), "./weight_c9/resnet18_btl_%s.pth" % (epoch))

        if (cur_acc >= acc) and (epoch >= (epochs - 10)):
            torch.save(net.state_dict(), "./weight_c9/resnet18_btl_best_%s_acc_%s.pth" % (epoch,cur_acc.cpu().detach().numpy()))
            acc = cur_acc

        if epoch % 20 == 0:
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
