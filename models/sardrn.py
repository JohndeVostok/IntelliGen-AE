import torch
from torch import nn
import onnx
import time
import add_size


def conv3x3(in_channels=64, out_channels=64, dilation=1, stride=1, padding=1):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, dilation=dilation, stride=stride, padding=padding)


class DRNBlock(nn.Module):
    def __init__(self, dilation1, dilation2):
        super(DRNBlock, self).__init__()
        self.conv1 = conv3x3(dilation=dilation1, padding=dilation1)
        self.conv2 = conv3x3(dilation=dilation2, padding=dilation2)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        return torch.add(x, out)


class DRN(nn.Module):
    def __init__(self):
        super(DRN, self).__init__()
        self.conv1 = conv3x3(in_channels=1)
        self.conv4 = conv3x3(dilation=4, padding=4)
        self.convx = conv3x3()
        self.conv7 = conv3x3(out_channels=1)
        self.relu = nn.ReLU()
        self.blk1 = DRNBlock(2, 3)
        self.blk2 = DRNBlock(3, 2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.blk1(out)
        out = self.conv4(out)
        out = self.relu(out)
        out = self.blk2(out)
        out = self.convx(out)
        out = self.relu(out)
        out = self.conv7(out)
        out = self.relu(out)
        return torch.sub(x, out)


def export_sardrn(path, batch_size, compiled):
    inputs = torch.randn(batch_size, 1, 512, 512)
    model = DRN()

    model_cuda = model.cuda().eval()
    input_cuda = inputs.cuda()

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
        torch.onnx.export(model_cuda, input_cuda, path, input_names=["input_0"], verbose=False, opset_version=9)
        model = onnx.load(path)
        add_size.add_value_info_for_constants(model)
        onnx.save(onnx.shape_inference.infer_shapes(model), path)


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    for compiled in [False, True]:
        export_sardrn("sardrn.bs1.onnx", 1, compiled)
        export_sardrn("sardrn.bs16.onnx", 16, compiled)
