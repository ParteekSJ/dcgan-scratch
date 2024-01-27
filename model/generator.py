from torch import nn
import ipdb, os
import torch
from torchinfo import summary


"""
ConvTranspose2D formulas
- H_out = [H_in - 1] * s - 2p + d * (f-1) + op + 1
- W_out = [W_in - 1] * s - 2p + d * (f-1) + op + 1
- Increases Spatial Dimensions

Conv2D formula
- H_out = [H_in + 2p + d(f-1) + 1 / s] + 1
- W_out = [W_in + 2p + d(f-1) + 1 / s] + 1
- Decreases Spatial Dimensions

"""


class Generator(nn.Module):
    def __init__(self, im_chan: int, z_dim: int, num_gen_filter: int):
        super(Generator, self).__init__()

        self.gen = nn.Sequential(
            self._generator_block(
                input_channels=z_dim,
                output_channels=num_gen_filter * 4,
                kernel_size=4,
                stride=2,
                padding=0,
            ),  # [B, 100, 1, 1] -> [B, 256, 4, 4]
            self._generator_block(
                input_channels=num_gen_filter * 4,
                output_channels=num_gen_filter * 2,
                kernel_size=3,
                stride=2,
                padding=1,
            ),  # [B, 256, 4, 4] -> [B, 128, 7, 7]
            self._generator_block(
                input_channels=num_gen_filter * 2,
                output_channels=num_gen_filter,
                kernel_size=4,
                stride=2,
                padding=1,
            ),  # [B, 128, 7, 7] -> [B, 256, 14, 14]
            self._generator_block(
                input_channels=num_gen_filter,
                output_channels=im_chan,
                kernel_size=4,
                stride=2,
                padding=1,
                final_layer=True,
            ),  # [B, 256, 14, 14] -> [B, 1, 28, 28]
        )

    def _generator_block(
        self,
        input_channels,
        output_channels,
        kernel_size=3,
        stride=2,
        padding=0,
        final_layer=False,
    ):
        # ipdb.set_trace()
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                # FROM PAPER: Using a bounded activation allowed the model to learn more
                # quickly to saturate and cover the color space of the training distribution.
                nn.Tanh(),  # rescales data between [-1, 1]
            )

    def forward(self, noise):
        # ipdb.set_trace()
        # 2D -> 4D, i.e., adding H & W dimensions for ConvTranspose2d
        generated_sample = self.gen(noise.unsqueeze(-1).unsqueeze(-1))  # [B, 100] -> [B, 100, 1, 1]
        return generated_sample


def create_noise_vector(n_samples: int, input_dim: int, device: str = "cpu"):
    return torch.randn(n_samples, input_dim).to(device)


if __name__ == "__main__":
    os.environ["IPDB_CONTEXT_SIZE"] = "7"
    ipdb.set_trace()
    gen = Generator(im_chan=1, z_dim=28, num_gen_filter=64)
    num_samples, z_dim = 100, 28

    sample_input = torch.randn(num_samples, z_dim)
    sample_output = gen(sample_input)
    print(f"{sample_output.shape=}")

    print(summary(model=gen, input_data=sample_input))
    """
    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    Generator                                [100, 1, 28, 28]          --
    ├─Sequential: 1-1                        [100, 1, 28, 28]          --
    │    └─Sequential: 2-1                   [100, 256, 4, 4]          --
    │    │    └─ConvTranspose2d: 3-1         [100, 256, 4, 4]          114,688
    │    │    └─BatchNorm2d: 3-2             [100, 256, 4, 4]          512
    │    │    └─ReLU: 3-3                    [100, 256, 4, 4]          --
    │    └─Sequential: 2-2                   [100, 128, 7, 7]          --
    │    │    └─ConvTranspose2d: 3-4         [100, 128, 7, 7]          294,912
    │    │    └─BatchNorm2d: 3-5             [100, 128, 7, 7]          256
    │    │    └─ReLU: 3-6                    [100, 128, 7, 7]          --
    │    └─Sequential: 2-3                   [100, 64, 14, 14]         --
    │    │    └─ConvTranspose2d: 3-7         [100, 64, 14, 14]         131,072
    │    │    └─BatchNorm2d: 3-8             [100, 64, 14, 14]         128
    │    │    └─ReLU: 3-9                    [100, 64, 14, 14]         --
    │    └─Sequential: 2-4                   [100, 1, 28, 28]          --
    │    │    └─ConvTranspose2d: 3-10        [100, 1, 28, 28]          1,024
    │    │    └─Tanh: 3-11                   [100, 1, 28, 28]          --
    ==========================================================================================


    As we can see we're constantly upsampling, i.e., increasing the SPATIAL DIMENSIONS of the image.
    - Spatial Dimensions INCREASE = [4,4] -> [7,7] -> [14,14] -> [28,28]
    - Output Channels DECREASE = 256 -> 128 -> 64 -> 1
    """
