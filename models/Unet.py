import torch
import torch.nn as nn

from .module import *

class UNet(nn.Module):
    def __init__(self, n_channels=1, conv_dim=32, n_classes=2, down_times=4, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.down_times = down_times
        self.bilinear = bilinear
        
        channel_list = [min((2**i)*conv_dim, 512) for i in range(down_times+1)]
        self.inc = DoubleConv(n_channels, channel_list[0])
        for idx in range(1, down_times+1):
            layer = Down(channel_list[idx-1], channel_list[idx])
            setattr(self, f'down{idx}', layer)
            
        channel_list = [min((2**i)*conv_dim, 512) for i in range(down_times, -1, -1)]
        for idx in range(down_times):
            layer = Up(channel_list[idx]+channel_list[idx+1], channel_list[idx+1])
            setattr(self, f'up{idx}', layer)

        self.outc = OutConv(conv_dim, n_classes)

    def forward(self, x):
        xs = []
        x = self.inc(x)
        xs.append(x)
        for idx in range(1, self.down_times+1):
            layer = getattr(self, f'down{idx}')
            x = layer(x)
            xs.append(x)
        
        for idx in range(self.down_times):
            layer = getattr(self, f'up{idx}')
            x = layer(x, xs[self.down_times-idx-1])
            
        logits = self.outc(x)
        return logits
    
if __name__ == '__main__':
    model = UNet()
    input = torch.randn([8, 1, 256, 256])
    output = model(input)
    print(output.shape)
#     print(model)