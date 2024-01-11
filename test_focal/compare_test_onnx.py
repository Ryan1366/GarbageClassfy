
# import cv2
# imgfile = 'imgfile.jpg'
# origimg1 = cv2.imread(imgfile)
# origimg = cv2.cvtColor(origimg1, cv2.COLOR_BGR2RGB)
# img_h, img_w = origimg.shape[:2]
# img = preprocess(origimg)
# img = img.astype(np.float32)
# img = img.transpose((2, 0, 1))

# img = np.expand_dims(img, axis=0)
# print(img.shape)

# ort_session = ort.InferenceSession('./yyd_10_24_indoorgarbage.onnx')
# res = (ort_session.run(None, {'data': img}))

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms, datasets, models

from tqdm import tqdm

from data_load import Date_Loader    
from model.resnet import resnet18

import onnxruntime as ort



train_dataset = Date_Loader("/home/yijiahe/you/dataset/garbage/clip_img/val.txt")  #
print("Train_img:",len(train_dataset))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1, drop_last = True)

net = resnet18(num_classes=4) ##scale 表示通道缩放比例 num_cls表示分类类别
net.load_state_dict(torch.load('./weights/resnet18_best_299_acc_99.97259.pth'))

print(net)
## ----打印计算量
'''
from torchstat import stat
net.eval()
stat(net, (3, 80, 80))
total = sum([param.nelement() for param in net.parameters()])
print("Number of parameters: %.2fM" % (total/1e6))
'''


if __name__ == '__main__':

    pt_result = open('./pt.txt','w')
    onnx_result = open('./onnx.txt','w')
    for data in train_loader:
        images, labels = data

        images, labels = Variable(images), Variable(labels)
        outputs = net(images)
        print("outputs:",outputs)
        pt_result.write(str(outputs))
        pt_result.write('\n')
        img = images.numpy().astype(np.float32)
        ort_session = ort.InferenceSession('./resnet18_best_299_acc_99.onnx')
        outs = (ort_session.run(None, {'input': img}))
        print("outs:",outs)
        onnx_result.write(str(outs))
        onnx_result.write('\n')



