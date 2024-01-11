import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset

from tqdm import tqdm

from data_load import Date_Loader    
from model.resnet import resnet18

# 配置参数
random_state = 1
torch.manual_seed(random_state)  # 设置随机数种子，确保结果可重复
torch.cuda.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)
np.random.seed(random_state)
# random.seed(random_state)

epochs = 300  
batch_size = 16
num_workers = 8  # 多线程的数目
use_gpu = torch.cuda.is_available()

pretrain = True

train_dataset = Date_Loader(path="/home/yijiahe/you/dataset/garbage_classfily/train.txt")  #
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle = True, num_workers=4, drop_last = True)


# 加载resnet18 模型
net = resnet18(num_classes = 4)
# net = ShuffleNet_v2(scale=1.0,num_cls=3) ##scale 表示通道缩放比例 num_cls表示分类类别

if pretrain:
    net.load_state_dict(torch.load('./weights/garbage_c4_80_pad.pth'))

if use_gpu:
    net = net.cuda()
print(net)

cirterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer,
                                                        T_max =  epochs) #  * iters 
net.train()

if __name__ == '__main__':

    for epoch in tqdm(range(epochs)):
        total_loss = 0.0
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        acc = 0
        for i, data in enumerate(train_loader):
            inputs, train_labels = data
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(train_labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(train_labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            # print("outputs:",outputs)
            _, train_predicted = torch.max(outputs.data, 1)
            # print("_, train_predicted:",_, train_predicted)
            train_correct += (train_predicted == labels.data).sum()
            # print("outputs, labels:",outputs, labels)
            print("labels:",labels)
            loss = cirterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            total_loss += running_loss
            train_total += train_labels.size(0)
        scheduler.step()
        print("Learning_rate:",optimizer.state_dict()['param_groups'][0]['lr'])
        print("Total loss:",total_loss)
        cur_acc = 100 * train_correct / train_total
        print('train %d epoch loss: %.3f  acc: %.3f ' % (
        epoch + 1, running_loss / train_total * batch_size, cur_acc))

        if epoch % 10 == 0 and epoch > 20:
            torch.save(net.state_dict(), "./weight/resnet18_%s.pth" % (epoch))

        if (cur_acc >= acc) and (epoch >= (epochs - 10)):
            torch.save(net.state_dict(), "./weight/resnet18_best_%s_acc_%s.pth" % (epoch,cur_acc.cpu().detach().numpy()))
            acc = cur_acc


'''
# ----- 打印模型的计算量和参数数量
from torchstat import stat
from torchvision.models import resnet50, resnet101, resnet152, resnext101_32x8d
 
net.eval()
stat(net, (3, 32, 32))

total = sum([param.nelement() for param in net.parameters()])
print("Number of parameters: %.2fM" % (total/1e6))
'''