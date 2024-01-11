import numpy as np
import torch
import torch.nn as nn
import cv2
from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision.transforms as transforms 

from tqdm import tqdm
from PIL import Image, ImageFile
from data_load import Date_Loader    
from torchvision.models import *
from labelsmoothing import CELoss

from sklearn.model_selection import  StratifiedKFold

class Date_Loader(Dataset):
    def __init__(self, imgs, labels, train_flag=True):

        self.imgs = imgs
        self.labels = labels

        self.train_flag = train_flag
        self.train_tf = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=(-10,10)),
                transforms.GaussianBlur(kernel_size=(5,9),sigma=(0.1,5)),
                transforms.ColorJitter(brightness=[.85,1.15],contrast=[.85,1.15],saturation=.0,hue=.0),
                transforms.ToTensor(),
            ])
        self.val_tf = transforms.Compose([
                transforms.ToTensor(),
            ])
        self.img_path = path
     
    def letter_reflect(self,img):

        h,w = img.shape[0],img.shape[1]
        scale = 112. / max(w, h)
        img = cv2.resize(img,(int(w * scale), int(h * scale)))
        if img.shape[0] >= img.shape[1]:
            bottom_size = 0
            left_size = 112 - img.shape[1]
        else:
            bottom_size = 112 - img.shape[0]
            left_size = 0
        img = cv2.copyMakeBorder(img,0,bottom_size,left_size,0,cv2.BORDER_REFLECT)
        img = cv2.resize(img,(112,112))
        return img

    def padding_black(self, img):

        w, h  = img.size
        scale = 112. / max(w, h)        
        img_fg = img.resize([int(x) for x in [w * scale, h * scale]]) 
        size_fg = img_fg.size
        size_bg = 112           
        img_bg = Image.new("RGB", (size_bg, size_bg))
        img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                              (size_bg - size_fg[1]) // 2))
        img = img_bg
        return img
        
    def __getitem__(self, index):

        img_path, label = self.imgs[index], self.labels[index]
        target = self.img_path.split(".")[0].split("/")[-1]
        image_dir= self.img_path.split(".")[0].replace(target,"images/")
        img_path = image_dir + img_path
        img = cv2.imread(img_path)        
        img = self.letter_reflect(img)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # cv转为pil，否则transform不能使用
        if self.train_flag:
            img = self.train_tf(img)
        else:
            img = self.val_tf(img)
        label = int(label)
        return img, label, img_path


    def __len__(self):
        return len(self.imgs)

def get_images(path):
    with open(path, 'r', encoding='utf-8') as f:
        imgs_info = f.readlines()
        imgs_list = list(map(lambda x:x.strip().split(' ')[0], imgs_info)) 
        labels_list = list(map(lambda x:x.strip().split(' ')[1], imgs_info)) 
    return imgs_list, labels_list

# 配置参数
random_state = 1
torch.manual_seed(random_state)  # 设置随机数种子，确保结果可重复
torch.cuda.manual_seed(random_state)
torch.cuda.manual_seed_all(random_state)
np.random.seed(random_state)


epochs = 100  
batch_size = 8
num_workers = 8  # 多线程的数目
use_gpu = torch.cuda.is_available()

net = mobilenet_v3_large(pretrained=True)
num_ftrs = net.classifier[3].in_features
net.classifier[3] = nn.Linear(num_ftrs, 9)


pretrain = 0
if pretrain:
    print("load pretrain model!")
    net.load_state_dict(torch.load('/home/yijiahe/you/garbage_c4/weight_c9/train_celoss_mobilev3/k_fold_1/mobile_best_90_train_acc_97.64994_test_acc_97.53886_loss_0.3987.pth'))

if use_gpu:
    net = net.cuda()

cirterion = CELoss(label_smooth=0.05, class_num=9)   
optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer, T_max =  epochs) #  * iters 

if __name__ == '__main__':

    skf = StratifiedKFold(n_splits=8) #5折
    path = "/home/yijiahe/you/dataset/cls9_1/label.txt"
    imgs_list, labels_list = get_images(path)

    for k, (train_idx, val_idx) in enumerate(skf.split(imgs_list, labels_list)):
        trainset, valset = np.array(imgs_list)[[train_idx]],np.array(imgs_list)[[val_idx]]
        traintag, valtag = np.array(labels_list)[[train_idx]],np.array(labels_list)[[val_idx]]
        train_dataset = Date_Loader(trainset, traintag)
        test_dataset = Date_Loader(valset, valtag, train_flag=False)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle = True, num_workers=4, drop_last = True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = True)

        for epoch in tqdm(range(epochs+1)):
            net.train()
            total_loss = 0.0
            running_loss = 0.0
            train_correct = 0
            train_total = 0
            target_loss = 0
            for i, data in enumerate(train_loader):
                train_inputs, train_labels, img_path = data
                if use_gpu:
                    train_inputs, labels = Variable(train_inputs.cuda()), Variable(train_labels.cuda())
                else:
                    train_inputs, labels = Variable(train_inputs), Variable(train_labels)
                optimizer.zero_grad()
                outputs = net(train_inputs)
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
            cur_acc = 100 * train_correct / train_total
            print('train %d epoch loss: %.3f  acc: %.3f ' % (
                epoch + 1, running_loss / train_total * batch_size, cur_acc))
            
            # if (epoch % 20 == 0) and (epoch >= (epochs - 30)) :
            #     net.eval()
            #     correct = 0
            #     test_loss = 0
            #     test_total = 0
            #     with torch.no_grad():
            #         for data in tqdm(test_loader):
            #             val_inputs, labels, img_path = data
            #             if use_gpu:
            #                 val_inputs, labels = Variable(val_inputs.cuda()), Variable(labels.cuda())
            #             else:
            #                 val_inputs, labels = Variable(val_inputs), Variable(labels)
            #             outputs = net(val_inputs)
            #             _, predict = torch.max(outputs.data, 1)
            #             loss = cirterion(outputs, labels)
            #             test_loss += loss
            #             test_total += labels.size(0)
            #             correct += (predict == labels.data).sum()
            #         torch.save(net.state_dict(), "./weight_c9/train_celoss_mobilev3/k_fold_1/resnet18_best_%s_train_acc_%s_test_acc_%s_loss_%s_k%s.pth" % (epoch,cur_acc.cpu().detach().numpy(),100 * correct / test_total, test_loss.sum() / test_total, k))
            #     print('test epoch loss: %.3f  acc: %.3f ' % ( test_loss.sum() / test_total, 100 * correct / test_total)) 
