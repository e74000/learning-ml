import mlx.core as mx
import mlx.nn as nn

class Discriminator(nn.Module):
    def __init__(self, ndf, nc):
        super().__init__()
        self.conv_1 = nn.Conv2d(nc,    ndf*1, 4, stride=2, padding=1, bias=False)
        self.conv_2 = nn.Conv2d(ndf*1, ndf*2, 4, stride=2, padding=1, bias=False)
        self.conv_3 = nn.Conv2d(ndf*2, ndf*4, 4, stride=2, padding=1, bias=False)
        self.conv_4 = nn.Conv2d(ndf*4, ndf*8, 4, stride=2, padding=1, bias=False)
        self.conv_5 = nn.Conv2d(ndf*8, 1,     4, stride=1, padding=0, bias=False)

        self.batch_norm_1 = nn.BatchNorm(ndf*2)
        self.batch_norm_2 = nn.BatchNorm(ndf*4)
        self.batch_norm_3 = nn.BatchNorm(ndf*8)

    def __call__(self, x):
        x = nn.leaky_relu(self.conv_1(x))
        x = nn.leaky_relu(self.batch_norm_1(self.conv_2(x)))
        x = nn.leaky_relu(self.batch_norm_2(self.conv_3(x)))
        x = nn.leaky_relu(self.batch_norm_3(self.conv_4(x)))
        x = nn.sigmoid(self.conv_5(x))
        x = mx.flatten(x)

        return x

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super().__init__()

        # Upsample layers (upsampling by factor of 2)
        self.upsample_1 = nn.Upsample(scale_factor=2, mode='linear')
        self.upsample_2 = nn.Upsample(scale_factor=2, mode='linear')
        self.upsample_3 = nn.Upsample(scale_factor=4, mode='linear')
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='linear')

        # Convolution layers after each upsampling
        self.conv_1 = nn.Conv2d(nz,    ngf*8, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_2 = nn.Conv2d(ngf*8, ngf*4, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_3 = nn.Conv2d(ngf*4, ngf*2, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_4 = nn.Conv2d(ngf*2, ngf*1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_5 = nn.Conv2d(ngf*1, nc,    kernel_size=3, stride=1, padding=1, bias=False)

        # Batch normalization layers
        self.batch_norm_1 = nn.BatchNorm(ngf*8)
        self.batch_norm_2 = nn.BatchNorm(ngf*4)
        self.batch_norm_3 = nn.BatchNorm(ngf*2)
        self.batch_norm_4 = nn.BatchNorm(ngf*1)

    def __call__(self, x):
        # First layer (no upsampling needed here)
        x = nn.relu(self.batch_norm_1(self.conv_1(x)))

        # Upsample and apply convolution + batch norm
        x = self.upsample_1(x)
        x = nn.relu(self.batch_norm_2(self.conv_2(x)))

        x = self.upsample_2(x)
        x = nn.relu(self.batch_norm_3(self.conv_3(x)))

        x = self.upsample_3(x)
        x = nn.relu(self.batch_norm_4(self.conv_4(x)))

        x = self.upsample_4(x)
        x = nn.tanh(self.conv_5(x))  # Final output (no batch norm, just tanh)

        return x