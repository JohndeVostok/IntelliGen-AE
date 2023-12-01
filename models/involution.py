import torch
import numpy as np
from torch import nn

class inv(nn.Module):
    # B: batch size, H: height, W: width
    # C: channel number, G: group number
    # K: kernel size, s: stride, r: reduction ratio
    def __init__(self, C, G, K, s = 1, r = 1, p = 0):
        super().__init__()
        self.C = C
        self.G = G
        self.K = K
        self.s = s
        self.p = p
        self.o = nn.AvgPool2d(s, s) if s > 1 else nn.Identity()
        self.reduce = nn.Conv2d(C, C//r, 1)
        self.span = nn.Conv2d(C//r, K*K*G, 1)
        self.unfold = nn.Unfold(K, dilation = 1, padding = p, stride = s)
    def forward(self, x):
        B = x.shape[0]
        C = self.C
        G = self.G
        K = self.K
        H = (x.shape[2] + 2 * self.p - K) // self.s + 1
        W = (x.shape[3] + 2 * self.p - K) // self.s + 1
        x_unfolded = self.unfold(x)
        x_unfolded = x_unfolded.view(B, G, C//G, K*K, H, W)
        #print(x_unfolded)
        kernel = self.span(self.reduce(self.o(x)))
        kernel = kernel.view(B, G, K*K, H, W).unsqueeze(2)
        out = torch.mul(kernel, x_unfolded).sum(dim = 3)
        out = out.view(B, C, H, W)
        return out

if __name__ == "__main__":
    model = inv(3, 1, 3, r = 3, p = 1)
    model = model.cuda()
    image = torch.randn(1, 3, 4, 4).cuda()
    print(image)
    print(model(image))