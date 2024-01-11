import torch
import onnx
# from shufflenetv2 import ShuffleNet_v2
import torchvision.models as models

torch_model = torch.load("/home/yijiahe/you/garbage_c4/weight_c9/resnet18_btl_80.pth") # pytorch模型加载

model = models.resnet18()
model.fc = torch.nn.Linear(512, 9)
model.load_state_dict(torch_model) 
batch_size = 1  #批处理大小
dummy_input = torch.ones(1, 3, 112, 112)

# #set the model to inference mode

export_onnx_file = "resnet18_btl_80.onnx"          # 目的ONNX文件名
torch.onnx.export(model.eval(), dummy_input, export_onnx_file, verbose=True, input_names=["input"], output_names=["output"], opset_version=10)
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
