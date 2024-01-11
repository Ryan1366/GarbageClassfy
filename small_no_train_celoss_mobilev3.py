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


epochs = 200  
batch_size = 8
num_workers = 8  # 多线程的数目
use_gpu = torch.cuda.is_available()


train_dataset = Date_Loader(path = "/home/yijiahe/you/dataset/small_no/train.txt") 
print("train_loader:",len(train_dataset))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle = True, num_workers=4, drop_last = True)


test_dataset = Date_Loader(path = "/home/yijiahe/you/dataset/small_no/val.txt",train_flag = False)
print("test_dataset:",len(test_dataset))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle = True)


net = mobilenet_v3_large(pretrained=True)
num_ftrs = net.classifier[3].in_features
net.classifier[3] = nn.Linear(num_ftrs, 7)


pretrain = 0
if pretrain:
    print("load pretrain model!")
    net.load_state_dict(torch.load('/home/yijiahe/you/garbage_c4/weight_c9/train_celoss_mobilev3/k_fold_1/mobile_best_96_train_acc_98.688_test_acc_98.57513_loss_tensor0.3717.pth'))

if use_gpu:
    net = net.cuda()

print("net:",net)

cirterion = CELoss(label_smooth=0.05, class_num=7)   
optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer,T_max =  epochs) #  * iters 

if __name__ == '__main__':

    for epoch in tqdm(range(epochs)):
        net.train()

        total_loss = 0.0
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        for i, data in enumerate(train_loader):
            inputs, train_labels,img_path = data
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(train_labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(train_labels)
            optimizer.zero_grad()

            outputs = net(inputs)
            _, train_predicted = torch.max(outputs.data, 1)
            train_correct += (train_predicted == labels.data).sum()
            loss = cirterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total_loss += running_loss
            train_total += train_labels.size(0)

        scheduler.step()
        print("Learning_rate:",optimizer.state_dict()['param_groups'][0]['lr'])
        print("Total loss:",total_loss)
        train_acc = 100 * train_correct / train_total
        print('train %d epoch loss: %.3f  acc: %.3f ' % (
        epoch + 1, running_loss / train_total * batch_size, train_acc))

        if (epoch % 10 == 0) or (epoch >= (epochs - 30)):
            net.eval()
            test_correct = 0
            test_loss = 0
            test_total = 0
            with torch.no_grad():
                for data in tqdm(test_loader):
                    val_inputs, labels, img_path = data
                    if use_gpu:
                        val_inputs, labels = Variable(val_inputs.cuda()), Variable(labels.cuda())
                    else:
                        val_inputs, labels = Variable(val_inputs), Variable(labels)
                    outputs = net(val_inputs)
                    _, predict = torch.max(outputs.data, 1)
                    loss = cirterion(outputs, labels)
                    test_loss += loss
                    test_total += labels.size(0)
                    test_correct += (predict == labels.data).sum()

            test_acc = 100 * test_correct / test_total
            torch.save(net.state_dict(), "./small_no/mobile_best_%s_train_acc_%s_test_acc_%s_loss_%s.pth" % (epoch, train_acc.cpu().detach().numpy(), test_acc.cpu().detach().numpy(), (test_loss.sum() / test_total).cpu().detach().numpy()))
            print('test epoch loss: %.3f  acc: %.3f ' % ( test_loss.sum() / test_total, test_acc)) 


