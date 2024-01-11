import torch
import onnx

# from shufflenetv2 import ShuffleNet_v2
from model.resnet import resnet18

torch_model = torch.load("./weights/resnet18_80.pth") # pytorch模型加载
model = resnet18(num_classes = 3)
# model.fc = torch.nn.Linear(2048, 4)
model.load_state_dict(torch_model) 
batch_size = 1  #批处理大小
dummy_input = torch.ones(1, 3, 224, 224)

# #set the model to inference mode
model.eval()

export_onnx_file = "resnet18_80.onnx"          # 目的ONNX文件名
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
