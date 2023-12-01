import torch
from transformers import BertLayer, BertConfig
import numpy as np
import onnx
import add_size
import time


class BertModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)[0]
        return x[0]


def export_bert(path, batch_size, compiled):
    inputs = torch.randn((batch_size, 512, 768))
    model = BertModel(BertConfig(vocab_size=32768, hidden_size=768,
                                 num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072))

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
        export_bert("bert.bs16.onnx", 1, compiled)
        export_bert("bert.bs16.onnx", 16, compiled)