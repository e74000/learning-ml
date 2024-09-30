import mlx.core as mx
import mlx.nn as nn

# from https://github.com/ml-explore/mlx-examples/blob/main/stable_diffusion/stable_diffusion/unet.py
def upsample_nearest(x, scale: int = 2):
    B, H, W, C = x.shape
    x = mx.broadcast_to(x[:, :, None, :, None, :], (B, H, scale, W, scale, C))
    x = x.reshape(B, H * scale, W * scale, C)
    return x

class UpsamplingConv2d(nn.Module):
    def __init__(self, i_channels, o_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(
            i_channels, o_channels, kernel_size, stride=stride, padding=padding
        )

    def __call__(self, x):
        x = self.conv(upsample_nearest(x))
        return x