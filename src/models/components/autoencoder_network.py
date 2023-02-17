#%%
import torch.nn as nn
import torch

class Shape_Decoder(nn.Module):
    def __init__(self, code_dim, output_channels, vae=False):
        super(Shape_Decoder, self).__init__()
        self.code_dim = code_dim
        self.dcnn = nn.Sequential(
            nn.ConvTranspose2d(code_dim, 256, 4, 1, 0),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, output_channels, 4, 2, 1),
            nn.Sigmoid() if vae else nn.Tanh()
        )

    def forward(self, z):
        out = self.dcnn(z.view(z.size(0), self.code_dim, 1, 1))
        return out


class Shape_Encoder(nn.Module):
    def __init__(self, code_dim, input_channels):
        super(Shape_Encoder, self).__init__()
        self.code_dim = code_dim
        self.dcnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, 4, 4, 1),    # output = 16x16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),      # 8x8
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, 2, 1),      # 4x4
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1),      # 2x2
            nn.ReLU(True),
            nn.Conv2d(256, code_dim, 2, 1, 0),    # 1x1
        )
        
    def forward(self, z):
        return self.dcnn(z).view(z.size(0), self.code_dim)


# #%%
# decoder = Shape_Decoder(2, 1, False)
# x = torch.randn(16, 2, 1, 1)
# for layer in decoder.dcnn:
#     x = layer(x)
#     print(x.shape)
# %%
