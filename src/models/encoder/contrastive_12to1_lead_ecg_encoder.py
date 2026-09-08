import numpy as np
import torch
import torch.nn as nn

from .ecg_founder import ECGFounder
from .resnet1d import ResNet1D


class Contrastive12To1LeadECGEncoder(nn.Module):
    """Contrasts a frozen 12-lead ECGFounder encoder against a trainable single-lead ResNet1D encoder."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.config.founder_configs.num_leads = len(self.config.founder_configs.input_channels)
        self.founder_encoder = ECGFounder(self.config.founder_configs)
        self.resnet_encoder = ResNet1D(self.config.resnet_configs)

        self.num_out_channels = (self.founder_encoder.num_out_channels, self.resnet_encoder.num_out_channels)

    def forward(self, batch_dict):
        waves = batch_dict["waves"].cuda()  # we want our model to be applicable to raw ecgs
        waves_ecgfounder = batch_dict["waves_filtered_normed"].cuda()  # use filtered and z-scored ECGs for ECGFounder

        founder_channel_data = []
        for channel in self.config.founder_configs.input_channels:
            founder_batch_idxs, founder_channel_idxs = np.where(batch_dict["channel_names"] == channel)
            founder_channel_data.append(waves_ecgfounder[founder_batch_idxs, founder_channel_idxs])
        founder_channel_data = torch.stack(founder_channel_data, dim=1)
        batch_dict["founder_representation"] = self.founder_encoder(founder_channel_data)

        resnet_channel_data = []
        for channel in self.config.resnet_configs.input_channels:
            if channel == "*":
                resnet_batch_idxs = np.arange(batch_dict["channel_names"].shape[0])
                resnet_channel_idxs = np.random.randint(0, batch_dict["channel_names"].shape[1], size=resnet_batch_idxs.shape[0])
            else:
                resnet_batch_idxs, resnet_channel_idxs = np.where(batch_dict["channel_names"] == channel)
            resnet_channel_data.append(waves[resnet_batch_idxs, resnet_channel_idxs])
        resnet_channel_data = torch.stack(resnet_channel_data, dim=1)
        batch_dict["resnet_representation"] = self.resnet_encoder(resnet_channel_data)

        return batch_dict
