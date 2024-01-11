import numpy as np
import torch
import os 
import cv2
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset

from tqdm import tqdm

from data_load import Date_Loader    
from model.resnet import resnet18
from torchvision.models import *

from labelsmoothing import CELoss

from confusionmartix import ConfusionMatrix

random_state = 1
torch.manual_seed(random_state)  
torch.cuda.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)
np.random.seed(random_state)

use_gpu = torch.cuda.is_available()


test_dataset = Date_Loader(path="/home/yijiahe/you/dataset/njn_c7/new_clip_image/val.txt",train_flag = False)#####测试数据集的地址

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

net = mobilenet_v3_large(pretrained=True)
num_ftrs = net.classifier[3].in_features
net.classifier[3] = nn.Linear(num_ftrs, 7)

net.load_state_dict(torch.load('/home/yijiahe/you/garbage_c4/njn_c7/mobile_best_260_train_acc_99.521164_test_acc_99.7735_loss_0.3231006.pth'))     ## 加载resnet18 模型

if use_gpu:
    net = net.cuda()
print("net",net)

# cirterion = CELoss(label_smooth=0.05, class_num=6)   

def make_dir(dir_name):
    if not os.path.exists(dir_name):  # os模块判断并创建
        os.mkdir(dir_name)


error_0 ='1008_c7/'
make_dir(error_0)

if __name__ == '__main__':

    # 模型测试
    correct = 0
    test_loss = 0.0
    test_total = 0
    test_total = 0

    '''
    ['瓜子','碎纸片','小米','烟头','纸片','绳子','其他']
    '''

    labels = ['0','1','2','3','4','5','6']
    confusion = ConfusionMatrix(num_classes=7, labels=labels)   
    net.eval()

    for data in tqdm(test_loader):
        image, labels, img_path= data
        '''       
        if labels== 2 or labels== 3:
            print("label",labels)
            continue
        '''
        if use_gpu:
            images, labels = Variable(image.cuda()), Variable(labels.cuda())
        else:
            images, labels = Variable(image), Variable(labels)
        outputs = net(images)
        _, predict = torch.max(outputs.data, 1)

        # loss = cirterion(outputs, labels)
        # test_loss += loss.item()
        # test_total += labels.size(0)
        
        correct += (predict == labels.data).sum() 
        outputs_1 = torch.softmax(outputs, dim=1)
        outputs_2 = torch.argmax(outputs_1, dim=1)


        
        if predict != labels.data:
            print("labels,outputs_2:",img_path)
            img = cv2.imread(img_path[0])
            save_img = img_path[0].split("/")[-1]
            prce = outputs_1.cpu().detach().numpy()[0][outputs_2.cpu().detach().numpy()]
            print("prce",prce)
            save_path = error_0 + "F"+str(predict.cpu().detach().numpy())+'_'+"T"+str(labels.data.cpu().detach().numpy())+'_prce_'+str(prce)+'_'+ save_img
            cv2.imwrite(save_path,img)
        '''
        if labels == 0 and outputs_2 == 1:
            img = cv2.imread(img_path[0])
            save_img = img_path[0].split("/")[-1]
            save_path = error_0 + save_img
            cv2.imwrite(save_path,img)

        elif labels == 1 and outputs_2 == 0:
            img = cv2.imread(img_path[0])
            save_img = img_path[0].split("/")[-1]
            save_path = error_1 + save_img
            cv2.imwrite(save_path,img)
        '''
        confusion.update(outputs_2.cpu().detach().numpy(), labels.cpu().detach().numpy())
    #print('test epoch loss: %.3f  acc: %.3f ' % ( test_loss / test_total, 100 * correct / test_total))
    confusion.plot()
    confusion.summary()
