import torch
import time
import onnx
import add_size


def export_shufflenet(path, batch_size, compiled):
    inputs = torch.randn(batch_size, 3, 224, 224)
    model = torch.hub.load('pytorch/vision:v0.10.0',
                           'shufflenet_v2_x1_0', pretrained=True)
    input_cuda = inputs.cuda()
    model_cuda = model.cuda().eval()

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
        export_shufflenet("shufflenet.bs1.onnx", 1, compiled)
        export_shufflenet("shufflenet.bs16.onnx", 16, compiled)
