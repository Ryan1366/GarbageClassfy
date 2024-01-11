import torch
import os
import glob
from torch.utils.data import Dataset
import random
import torchvision.transforms as transforms 
import cv2

import numpy as np

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Date_Loader(Dataset):
    def __init__(self, path, train_flag=True):
        self.imgs_info = self.get_images(path)
        self.train_flag = train_flag
        
        self.train_tf = transforms.Compose([
                #transforms.Resize(112),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(degrees=(-10,10)),
                transforms.GaussianBlur(kernel_size=(5,9),sigma=(0.1,5)),
                transforms.ColorJitter(brightness=[.85,1.15],contrast=[.85,1.15],saturation=.0,hue=.0),
                # transforms.RandomVerticalFlip(),
                #transforms.RandomGrayscale(p=0.1)
                transforms.ToTensor(),
            ])
        self.val_tf = transforms.Compose([
                #transforms.Resize(112),
                transforms.ToTensor(),
            ])
        self.img_path = path


    def get_images(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x:x.strip().split(' '), imgs_info)) 
            # print("imgs_info:",imgs_info)
        return imgs_info
     
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
        
    # def __getitem__(self, index):

    #     img_path, label = self.imgs_info[index]
    #     target = self.img_path.split(".")[0].split("/")[-1]
    #     image_dir= self.img_path.split(".")[0].replace(target,"images/")
    #     img_path = image_dir + img_path
    #     img = Image.open(img_path)
    #     img = img.convert('RGB')
        
    #     img = self.padding_black(img)
    #     if self.train_flag:
    #         img = self.train_tf(img)
    #     else:
    #         img = self.val_tf(img)
    #     label = int(label)
    #     return img, label, img_path

    def __getitem__(self, index):

        img_path, label = self.imgs_info[index]
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
        # 适配 bi-tempered-loss
        # bitemperedlosslabel = np.array([0,0,0,0,0,0,0,0,0])
        # bitemperedlosslabel[label] = 1
        # return img, label, img_path, bitemperedlosslabel
        return img, label, img_path


    def __len__(self):
        return len(self.imgs_info)
 
    
if __name__ == "__main__":
    train_dataset = Date_Loader("/home/yijiahe/you/dataset/cls9/label.txt", True)
    print("数据个数：", len(train_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=1, 
                                               shuffle=True)
    for image, label,img_path in train_loader:
        # print(image.shape)
        # print(label)
        print("img_path:",img_path)
