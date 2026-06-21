import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int, dilation: int
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        pad = (self.kernel_size - 1) * self.dilation

        out = F.pad(x, (pad, 0))
        out = F.relu(self.conv1(out))

        out = F.pad(out, (pad, 0))
        out = self.conv2(out)
        res = x if self.downsample is None else self.downsample(x)
        return F.relu(out + res)


class TCNModel(nn.Module):
    def __init__(
        self,
        num_inputs: int = 1,
        output_channels: list[int] = [16, 16, 16],
        kernel_size: int = 2,
    ):

        super().__init__()
        layers = []
        in_ch = num_inputs
        for i, ch in enumerate(output_channels):
            dilation = 2**i
            layers.append(ResidualBlock(in_ch, ch, kernel_size, dilation))
            in_ch = ch
        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(output_channels[-1], 1)

    def forward(self, x):
        out = self.tcn(x)
        out = out[:, :, -1]
        out = self.linear(out)
        return out.squeeze(-1)
