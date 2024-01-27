from torch import nn
import ipdb, os
import torch
from torchinfo import summary

"""
- LeakyReLU helps in faster convergence if the output requires both positive and negative values.
"""


class Discriminator(nn.Module):
    def __init__(self, im_chan, num_disc_filter):
        super(Discriminator, self).__init__()

        # Depth keeps increasing as we go through the network.
        self.disc = nn.Sequential(
            self._discriminator_block(
                input_channels=im_chan,
                output_channels=num_disc_filter,
                kernel_size=4,
                stride=2,
                padding=1,
            ),  # [100, 1, 28, 28] -> [100, 64, 14, 14]
            self._discriminator_block(
                input_channels=num_disc_filter,
                output_channels=num_disc_filter * 2,
                kernel_size=4,
                stride=2,
                padding=1,
            ),  # [100, 64, 14, 14] -> [100, 128, 7, 7]
            self._discriminator_block(
                input_channels=num_disc_filter * 2,
                output_channels=num_disc_filter * 4,
                kernel_size=3,
                stride=2,
                padding=1,
            ),  # [100, 128, 7, 7] -> [100, 256, 4, 4]
            self._discriminator_block(
                input_channels=num_disc_filter * 4,
                output_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                final_layer=True,
            ),  # [100, 256, 4, 4] -> [100, 1, 1, 1]
        )

    def _discriminator_block(
        self,
        input_channels,
        output_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        final_layer=False,
    ):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features=output_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                ),
                # if Sigmoid, use BCELoss(), if no Sigmoid, use BCEWithLogitsLoss()
                # nn.Sigmoid()
            )

    def forward(self, image):
        disc_pred = self.disc(image)  # [B, C, H, W]
        return disc_pred.view(-1, 1)
        # return disc_pred.view(-1, 1).squeeze(1)
        # return disc_pred.reshape(len(disc_pred), -1)


if __name__ == "__main__":
    os.environ["IPDB_CONTEXT_SIZE"] = "7"
    ipdb.set_trace()
    disc = Discriminator(im_chan=1, num_disc_filter=64)

    sample_input = torch.randn(10, 1, 28, 28)
    sample_output = disc(sample_input)
    print(f"{sample_output.shape=}")

    print(summary(model=disc, input_data=sample_input))

    """
    ==========================================================================================
    Layer (type:depth-idx)                   Output Shape              Param #
    ==========================================================================================
    Discriminator                            [10]                      --
    ├─Sequential: 1-1                        [10, 1, 1, 1]             --
    │    └─Sequential: 2-1                   [10, 64, 14, 14]          --
    │    │    └─Conv2d: 3-1                  [10, 64, 14, 14]          1,024
    │    │    └─BatchNorm2d: 3-2             [10, 64, 14, 14]          128
    │    │    └─LeakyReLU: 3-3               [10, 64, 14, 14]          --
    │    └─Sequential: 2-2                   [10, 128, 7, 7]           --
    │    │    └─Conv2d: 3-4                  [10, 128, 7, 7]           131,072
    │    │    └─BatchNorm2d: 3-5             [10, 128, 7, 7]           256
    │    │    └─LeakyReLU: 3-6               [10, 128, 7, 7]           --
    │    └─Sequential: 2-3                   [10, 256, 4, 4]           --
    │    │    └─Conv2d: 3-7                  [10, 256, 4, 4]           294,912
    │    │    └─BatchNorm2d: 3-8             [10, 256, 4, 4]           512
    │    │    └─LeakyReLU: 3-9               [10, 256, 4, 4]           --
    │    └─Sequential: 2-4                   [10, 1, 1, 1]             --
    │    │    └─Conv2d: 3-10                 [10, 1, 1, 1]             4,096
    │    │    └─Sigmoid: 3-11                [10, 1, 1, 1]             --
    ==========================================================================================
    """
