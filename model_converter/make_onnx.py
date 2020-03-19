import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx 
from onnx import load
import onnxsim 

dummy_input = torch.randn(1, 3, 256, 256)
net=torch.load("model.pth")
net.cuda()
dummy_output=net(dummy_input.cuda())
net.eval()
torch.onnx.export(net.cuda(), dummy_input.cuda(),
                  "fr_net.onnx",
                  export_params=True,
                  input_names=['input_img'],
                  output_names=['pos_img'])

# Preprocessing: load the model to be optimized.
model_path = "fr_net.onnx"
original_model = load(model_path)        
# print('The model before optimization:\n{}'.format(original_model))
simplified_model=onnxsim.simplify(original_model)
onnx.save(simplified_model,"frNet_model.onnx")

