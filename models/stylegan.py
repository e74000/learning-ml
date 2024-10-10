import mlx.core as mx
import mlx.nn as nn

class AdaIN(nn.Module):
    def __init__(self, w_dim, num_features):
        super().__init__()
        self.gamma = nn.Linear(w_dim, num_features)
        self.beta  = nn.Linear(w_dim, num_features)
        self.instance_norm = nn.InstanceNorm(dims=num_features)

    def __call__(self, x, w):
        x = self.instance_norm(x)

        gamma = self.gamma(w).reshape(-1, 1, 1, x.shape[-1])
        beta = self.beta(w).reshape(-1, 1, 1, x.shape[-1])

        return x * gamma + beta

class SynthesisBlock(nn.Module):
    def __init__(self, i_c, o_c, w_dim, use_upsample=True):
        super().__init__()

        if use_upsample:
            self.use_upsample = True
            self.upsample = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)

        self.conv_1 = nn.Conv2d(i_c, o_c, kernel_size=3, stride=1, padding=1)
        self.conv_2 = nn.Conv2d(o_c, o_c, kernel_size=3, stride=1, padding=1)

        self.adain_1 = AdaIN(w_dim, o_c)
        self.adain_2 = AdaIN(w_dim, o_c)

        mx.ones()

    def __call__(self, x, w):
        if self.use_upsample:
            x = self.upsample(x)

        x = self.conv_1(x)
        # add noise?
        x = self.adain_1(x, w)
        x = self.conv_2(x)
        # add noise?
        x = self.adain_2(x, w)

        return x

class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=3):
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
    def __init__(self, ndf=64, nc=3):
        super().__init__()