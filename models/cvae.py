import math

import mlx.core as mx
import mlx.nn as nn

from conv_utils import UpsamplingConv2d

class Encoder(nn.Module):
    def __init__(self, n_latent, image_shape, max_n_filters):
        super().__init__()

        n_filters_1 = max_n_filters // 8
        n_filters_2 = max_n_filters // 4
        n_filters_3 = max_n_filters // 2
        n_filters_4 = max_n_filters // 1

        self.conv_1 = nn.Conv2d(image_shape[-1], n_filters_1, 4, stride=2, padding=1)
        self.conv_2 = nn.Conv2d(n_filters_1,     n_filters_2, 4, stride=2, padding=1)
        self.conv_3 = nn.Conv2d(n_filters_2,     n_filters_3, 4, stride=2, padding=1)
        self.conv_4 = nn.Conv2d(n_filters_3,     n_filters_4, 4, stride=2, padding=1)

        self.batch_norm_1 = nn.BatchNorm(n_filters_1)
        self.batch_norm_2 = nn.BatchNorm(n_filters_2)
        self.batch_norm_3 = nn.BatchNorm(n_filters_3)
        self.batch_norm_4 = nn.BatchNorm(n_filters_4)

        output_shape = [n_filters_4] + [d // 16 for d in image_shape[:-1]]
        flattened_dim = math.prod(output_shape)

        self.proj_mu      = nn.Linear(flattened_dim, n_latent)
        self.proj_log_var = nn.Linear(flattened_dim, n_latent)

    def __call__(self, x):
        x = nn.leaky_relu(self.batch_norm_1(self.conv_1(x)))
        x = nn.leaky_relu(self.batch_norm_2(self.conv_2(x)))
        x = nn.leaky_relu(self.batch_norm_3(self.conv_3(x)))
        x = nn.leaky_relu(self.batch_norm_4(self.conv_4(x)))
        x = mx.flatten(x, 1)

        mu = self.proj_mu(x)
        log_var = self.proj_log_var(x)
        sigma = mx.exp(log_var * 0.5)

        eps = mx.random.normal(sigma.shape)

        z = eps * sigma + mu

        return z, mu, log_var

class Decoder(nn.Module):
    def __init__(self, n_latent, image_shape, max_n_filters):
        super().__init__()

        self.n_latent = n_latent
        n_channels = image_shape[-1]
        self.max_n_filters = max_n_filters

        n_filters_1 = max_n_filters // 1
        n_filters_2 = max_n_filters // 2
        n_filters_3 = max_n_filters // 4
        n_filters_4 = max_n_filters // 8

        self.input_shape = [d // 16 for d in image_shape[:-1]] + [n_filters_1]
        flattened_dim = math.prod(self.input_shape)

        self.lin1 = nn.Linear(n_latent, flattened_dim)

        self.up_conv_1 = UpsamplingConv2d(n_filters_1, n_filters_2, 3, 1, 1)
        self.up_conv_2 = UpsamplingConv2d(n_filters_2, n_filters_3, 3, 1, 1)
        self.up_conv_3 = UpsamplingConv2d(n_filters_3, n_filters_4, 3, 1, 1)
        self.up_conv_4 = UpsamplingConv2d(n_filters_4, n_channels,  3, 1, 1)

        self.batch_norm_1 = nn.BatchNorm(n_filters_2)
        self.batch_norm_2 = nn.BatchNorm(n_filters_3)
        self.batch_norm_3 = nn.BatchNorm(n_filters_4)

    def __call__(self, z):
        x = self.lin1(z)

        x = x.reshape(-1, self.input_shape[0], self.input_shape[1], self.max_n_filters)

        x = nn.relu(self.batch_norm_1(self.up_conv_1(x)))
        x = nn.relu(self.batch_norm_2(self.up_conv_2(x)))
        x = nn.relu(self.batch_norm_3(self.up_conv_3(x)))
        x = nn.sigmoid(self.up_conv_4(x))

        return x

class CVAE(nn.Module):
    def __init__(self, n_latent, input_shape, max_n_filters):
        super().__init__()
        self.n_latent = n_latent
        self.encoder = Encoder(n_latent, input_shape, max_n_filters)
        self.decoder = Decoder(n_latent, input_shape, max_n_filters)

    def __call__(self, x):
        z, mu, log_var = self.encoder(x)
        x = self.decoder(z)
        return x, mu, log_var

    def encode(self, x):
        return self.encoder(x)[0]

    def decode(self, z):
        return self.decoder(z)