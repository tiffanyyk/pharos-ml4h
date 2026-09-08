import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, downsample, in_channels, out_channels, kernel_size=16, stride=1, padding="same"):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, downsample, padding=0, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.kernel_size = kernel_size
        self.downsample = downsample
        self.pool = nn.MaxPool1d(kernel_size=downsample, stride=downsample)
        self.identity_layer = None
        if in_channels != out_channels:
            self.identity_layer = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

        self.manual_padding = ((self.kernel_size + 1 - self.downsample) // 2, (self.kernel_size - self.downsample) // 2)

    def forward(self, x, residual):
        residual = self.pool(residual)
        residual = residual if self.identity_layer is None else self.identity_layer(residual)
        out = F.relu(self.bn1(self.conv1(x)))
        # Manual same padding, Pytorch same padding doesn't work with stride > 1
        out = self.conv2(F.pad(out, self.manual_padding))
        # Add skip before bn&act https://github.com/broadinstitute/ml4h/blob/master/model_zoo/PCLR/build_model.py#L249
        #   TODO: make preactivation into an option
        out += residual
        residual = out
        out = F.relu(self.bn2(out))
        return out, residual


class ResNet1D(nn.Module):
    def __init__(self, config):
        super(ResNet1D, self).__init__()
        self.config = config
        self.config.num_input_channels = 12 if self.config.get("use_12_leads_and_mask", False) else len(self.config.input_channels)
        self.num_out_channels = self.config.num_filters[-1]

        kernel_size = self.config.get("kernel_size", 16)
        self.conv1 = nn.Conv1d(self.config.num_input_channels, self.config.num_filters[0], kernel_size=kernel_size, padding="same", bias=False)
        self.bn1 = nn.BatchNorm1d(self.config.num_filters[0])

        self.num_res_blocks = len(self.config.num_filters) - 1  # last one is representation dim
        res_blocks = []
        res_block_kernel_size = self.config.get("res_block_kernel_size", 16)
        for block in range(self.num_res_blocks):
            res_blocks.append(ResidualBlock(self.config.pool_kernel_size_res_block, self.config.num_filters[block],
                                            self.config.num_filters[block + 1], kernel_size=res_block_kernel_size))
        self.res_blocks = nn.ModuleList(res_blocks)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        if self.config.get("pretrained_weights") is not None:
            self.load_pretrained_pytorch_weights(self.config.pretrained_weights)

        if self.config.get("freeze_weights"):
            for param in self.parameters():
                param.requires_grad = False

    def load_pretrained_pytorch_weights(self, file):
        checkpoint = torch.load(file)
        missing_keys, unexpected_keys = self.load_state_dict(checkpoint["model_state"], strict=False)
        if len(missing_keys) > 0:
            self.load_state_dict(checkpoint["model_state"], strict=False)
        key_start_idx = len(f"encoder.{self.config.pretrained_key}.")
        weights = {k[key_start_idx:]: v for (k, v) in checkpoint["model_state"].items() if self.config.pretrained_key in k}
        self.load_state_dict(weights)

    def forward(self, x):
        if type(x) is dict:
            batch_dict = x
            waves = batch_dict["waves"].cuda()

            channel_data = []
            for channel in self.config.input_channels:
                ecg_batch_idxs, ecg_channel_idxs = np.where(batch_dict["channel_names"] == channel)
                channel_data.append(waves[ecg_batch_idxs, ecg_channel_idxs])
            channel_data = torch.stack(channel_data, dim=1)
            batch_dict["ecg_representation"] = self.resnet_forward(channel_data)
            return batch_dict
        else:
            return self.resnet_forward(x)

    def resnet_forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        residual = x
        for res_block in self.res_blocks:
            x, residual = res_block(x, residual)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        return x


if __name__ == "__main__":
    from easydict import EasyDict
    config = EasyDict({
        "input_channels": [ "I" ],
        "num_filters": [ 64, 256 ],
        "pool_kernel_size_res_block": 12,
        "kernel_size": 17,
        "res_block_kernel_size": 17,
    })
    model = ResNet1D(config)
