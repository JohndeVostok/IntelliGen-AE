import torch
from involution import inv
from torch import nn
import time
import numpy as np
import onnx
import add_size

class BottleNeckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(BottleNeckBlock, self).__init__()
        
        mid_channels = out_channels // 4
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = inv(mid_channels, G = 1, K = 3, s = stride, p = 1)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace = True)
        self.stride = stride
        
        if downsample is not None:
            self.downsample = downsample
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace = True)
            )
    
    def forward(self, x):
        res = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            res = self.downsample(res)
        
        out += res
        out = self.relu(out)
        return out


class RedNet_50(nn.Module):
    def __init__(self):
        super(RedNet_50, self).__init__()
        self.layer0_conv = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.layer0_bn = nn.BatchNorm2d(64)
        self.layer0_pool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.layer1 = self._make_layer(3, 64, 256, 1)
        self.layer2 = self._make_layer(4, 256, 512, 2)
        self.layer3 = self._make_layer(6, 512, 1024, 2)
        self.layer4 = self._make_layer(3, 1024, 2048, 2)
        
        self.average_pool = nn.AvgPool2d(3, stride = 1, padding = 1)
        self.fc = nn.Linear(7 * 7 * 2048, 1000)
        
    def forward(self, x):
        out = self.layer0_conv(x)
        out = self.layer0_bn(out)
        out = self.layer0_pool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.average_pool(out)
        out = out.view(-1, 7 * 7 * 2048)
        out = self.fc(out)
        return out
        
    def _make_layer(self, block_num, in_channels, out_channels, stride = 1):
        layers = []
        block = BottleNeckBlock(in_channels, out_channels, stride = stride)
        layers.append(block)
        
        for i in range(block_num - 1):
            block = BottleNeckBlock(out_channels, out_channels, stride = 1)
            layers.append(block)
            
        return nn.Sequential(*layers)

def export_rednet(path, batch_size, compiled):
    image = torch.randn(batch_size, 3, 224, 224).cuda()
    model = RedNet_50()
    model_cuda = model.cuda()
    input_cuda = image.cuda()
    if compiled:
        model_cuda = torch.compile(model_cuda)
    
    with torch.no_grad():
        for _ in range(128):
            _ = model_cuda(input_cuda)
        torch.cuda.synchronize()
        time_st = time.time()
        for _ in range(128):
            _ = model_cuda(input_cuda)
        torch.cuda.synchronize()
        time_ed = time.time()
        print("PyTorch Time: ", (time_ed - time_st) / 128 * 1000, "ms")

    # with torch.no_grad():
    #     model_cuda_jit = torch.jit.trace(model_cuda, input_cuda)
    #     for _ in range(128):
    #         _ = model_cuda_jit(input_cuda)
    #     torch.cuda.synchronize()
    #     time_st = time.time()
    #     for _ in range(128):
    #         _ = model_cuda_jit(input_cuda)
    #     torch.cuda.synchronize()
    #     time_ed = time.time()
    #     print("TorchScript Time: ", (time_ed - time_st) / 128 * 1000, "ms")


    if not compiled:
        param = input_cuda
        torch.onnx.export(model_cuda, param, path, input_names=["input_0"], verbose = False, opset_version = 11)
        model = onnx.load(path)
        add_size.add_value_info_for_constants(model)
        onnx.save(onnx.shape_inference.infer_shapes(model), path)

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    for compiled in [False, True]:
        export_rednet("rednet50.bs16.onnx", 1, compiled)
        #export_rednet("rednet50.bs16.onnx", 16, compiled)
