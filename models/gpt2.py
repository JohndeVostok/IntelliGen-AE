import torch
import numpy as np
import onnx
import time
import add_size

class GPT2Attention(torch.nn.Module):
    def __init__(self, batch_size, hidden_size, num_heads, index):
        super().__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.past_key = torch.rand((batch_size, index-1, hidden_size)).cuda()
        self.past_value = torch.rand((batch_size, index-1, hidden_size)).cuda()

        self.c_attn = torch.nn.Linear(hidden_size, hidden_size * 3)
        self.c_proj = torch.nn.Linear(hidden_size, hidden_size)

    def _split_heads(self, tensor, num_heads, attn_head_size):
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    def forward(self, x):
        query, key, value = self.c_attn(x).split(self.hidden_size, dim=2)
        key = torch.cat((self.past_key, key), dim=1)
        value = torch.cat((self.past_value, value), dim=1)

        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        x = torch.nn.functional.softmax(torch.matmul(query, key.transpose(-1, -2)) / np.sqrt(self.head_dim), dim=-1)
        x = torch.matmul(x, value)
        x = x.transpose(1, 2).contiguous().view(self.batch_size, self.hidden_size)
        x = self.c_proj(x)
        x = x.view(self.batch_size, 1, self.hidden_size)
        return x


class GPT2MLP(torch.nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(hidden_size, hidden_size * 4)
        self.fc2 = torch.nn.Linear(hidden_size * 4, hidden_size)
    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


class GPT2Block(torch.nn.Module):
    def __init__(self, batch_size, hidden_size, num_heads, index):
        super().__init__()
        self.ln_1 = torch.nn.LayerNorm(hidden_size, eps=1e-5)
        self.attn = GPT2Attention(batch_size=batch_size, hidden_size=hidden_size, num_heads=num_heads, index=index)
        self.ln_2 = torch.nn.LayerNorm(hidden_size, eps=1e-5)
        self.mlp = GPT2MLP(hidden_size=hidden_size)
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT2Model(torch.nn.Module):
    def __init__(self, batch_size, hidden_size, num_layers, num_heads, index):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [GPT2Block(batch_size=batch_size, hidden_size=hidden_size, num_heads=num_heads, index=index) for _ in range(num_layers)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def export_gpt2(path, batch_size, num_layers, index, compiled):
    inputs = torch.randn((batch_size, 1, 768))
    model = GPT2Model(batch_size=batch_size, hidden_size=768, num_layers=num_layers, num_heads=12, index=index)

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
    # export_gpt2("gpt2.onnx", 128, 1, 64)
    for compiled in [False, True]:
        export_gpt2("gpt2.onnx", 128, 12, 64, compiled)
    # export_gpt2("gpt2.onnx", 128, 12, 64, True)

    # export_gpt2("gpt2.l12i64.onnx", 128, 12, 64)
    # export_gpt2("gpt2.l1i64.onnx", 128, 1, 64)
    # export_bert("bert.bs16.onnx", 16)
    # export_gpt2("gpt2.l1i1.onnx", 32, 1, 1)
    # export_gpt2("gpt2.l1i2.onnx", 32, 1, 2)
    # export_gpt2("gpt2.l1i4.onnx", 32, 1, 4)
    # export_gpt2("gpt2.l1i8.onnx", 32, 1, 8)
    # export_gpt2("gpt2.l1i16.onnx", 32, 1, 16)
    # export_gpt2("gpt2.l1i32.onnx", 32, 1, 32)
    # export_gpt2("gpt2.l1i64.onnx", 32, 1, 64)
    # export_gpt2("gpt2.l1i128.onnx", 32, 1, 128)
    # export_gpt2("gpt2.l1i256.onnx", 32, 1, 256)
    # export_gpt2("gpt2.l1i512.onnx", 32, 1, 512)
    
