import torch
import onnx

import numpy as np
import torch.nn as nn
import onnxruntime as ort

from torchvision.models import *

torch_model = torch.load("/home/yijiahe/you/garbage_c4/mobile_best_299_train_acc_99.72682_test_acc_99.70352_loss_0.24565126.pth") # pytorch模型加载
model = mobilenet_v3_large(pretrained=True)
num_ftrs = model.classifier[3].in_features
model.classifier[3] = nn.Linear(num_ftrs, 3) # 类别数量

model.load_state_dict(torch_model) 
dummy_input = torch.ones(1, 3, 112, 112)  # 输入图像大小

#测试pt模型
model.eval()
outputs = model(dummy_input)
print("pt_out:",outputs[0])

export_onnx_file = "MobileV3_V1_20231208_C3.onnx"          # 目的ONNX文件名
torch.onnx.export(model.eval(), dummy_input, export_onnx_file, verbose=False, input_names=["input"], output_names=["output"], opset_version=10)
print("======================== convert onnx Finished! .... ")

try:
    import onnxsim
    onnx_model = onnx.load(export_onnx_file)
    print('\nStarting to simplify ONNX...')
    sim_onnx_model, check = onnxsim.simplify(onnx_model)
    assert check, 'assert check failed'
except Exception as e:
    print(f'Simplifier failure: {e}')
onnx.save(sim_onnx_model,export_onnx_file)
print("Success to Simplify !!!")


#测试onnx
input_onnx = dummy_input.numpy().astype(np.float32)
ort_session = ort.InferenceSession('./MobileV3_V1_20231208_C3.onnx')
outs = (ort_session.run(None, {'input': input_onnx}))
print("onnx_out:",outs[0][0])
