import torch

import numpy as np
import torch.nn as nn
import onnxruntime as ort

from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision.models import *
from tqdm import tqdm
from data_load import Date_Loader

train_dataset = Date_Loader("/home/yijiahe/you/dataset/small_large_yes/val.txt",train_flag = False)  #
print("Train_img:",len(train_dataset))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1, drop_last = True)


net = mobilenet_v3_large(pretrained=True)
num_ftrs = net.classifier[3].in_features
net.classifier[3] = nn.Linear(num_ftrs, 7)  #类别数量
net.load_state_dict(torch.load('/home/yijiahe/you/garbage_c4/small_large_yes/mobile_best_299_train_acc_98.81966_test_acc_97.91305_loss_0.3600538.pth'))
net.eval()

print(net)

if __name__ == '__main__':

    pt_result = open('./pt.txt','w')
    onnx_result = open('./onnx.txt','w')
    for data in train_loader:
        images, labels, img_path = data

        images, labels = Variable(images), Variable(labels)
        outputs = net(images)
        print("outputs:",outputs)
        pt_result.write(str(outputs))
        pt_result.write('\n')
        img = images.numpy().astype(np.float32)
        ort_session = ort.InferenceSession('./0328_small_large_yes.onnx')
        outs = (ort_session.run(None, {'input': img}))
        print("outs:",outs)
        onnx_result.write(str(outs))
        onnx_result.write('\n')
