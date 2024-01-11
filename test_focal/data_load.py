import torch
from PIL import Image
import os
import glob
from torch.utils.data import Dataset
import random
import torchvision.transforms as transforms 
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Date_Loader(Dataset):
    def __init__(self, path, train_flag=True):
        self.imgs_info = self.get_images(path)
        self.train_flag = train_flag
        
        self.train_tf = transforms.Compose([
                transforms.Resize(80),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),

            ])
        self.val_tf = transforms.Compose([
                transforms.Resize(80),
                transforms.ToTensor(),
            ])
        self.img_path = path


    def get_images(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            imgs_info = f.readlines()
            imgs_info = list(map(lambda x:x.strip().split(' '), imgs_info)) 
            # print("imgs_info:",imgs_info)
        return imgs_info
     
    def padding_black(self, img):

        w, h  = img.size

        scale = 80. / max(w, h)         ###80
        img_fg = img.resize([int(x) for x in [w * scale, h * scale]])
        size_fg = img_fg.size
        size_bg = 80           ###80

        img_bg = Image.new("RGB", (size_bg, size_bg))

        img_bg.paste(img_fg, ((size_bg - size_fg[0]) // 2,
                              (size_bg - size_fg[1]) // 2))

        img = img_bg
        return img
        
    def __getitem__(self, index):

        img_path, label = self.imgs_info[index]
        target = self.img_path.split(".")[0].split("/")[-1]
        image_dir= self.img_path.split(".")[0].replace(target,"images/")
        img_path = image_dir + img_path
        img = Image.open(img_path)
        img = img.convert('RGB')
        # img=img.resize((80,80),Image.BILINEAR)  
        img = self.padding_black(img)
        if self.train_flag:
            img = self.train_tf(img)
        else:
            img = self.val_tf(img)
        label = int(label)
        return img, label
 
    def __len__(self):
        return len(self.imgs_info)
 
    
if __name__ == "__main__":
    train_dataset = Date_Loader("/home/yijiahe/you/dataset/garbage_classfily/val.txt", True)
    print("数据个数：", len(train_dataset))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=1, 
                                               shuffle=True)
    for image, label in train_loader:
        print(image.shape)
        print(label)
