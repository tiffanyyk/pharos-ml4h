import random

import numpy as np
import torch
import torch.nn as nn
from src.models.utils.create_networks import create_mlp
from src.models.encoder.ecg_founder import ECGFounder
from .resnet1d import ResNet1D


class ECGPairPADeltatEncoder(nn.Module):
    def __init__(self, config):
        super(ECGPairPADeltatEncoder, self).__init__()
        self.config = config

        self.num_out_channels = 0
        if self.config.ecg_configs.get("use_ecg", True):
            if self.config.ecg_configs.get("use_foundation", False):
                if self.config.ecg_configs.get("foundation_model", "ecg_founder") == "ecg_founder":
                    self.ecg_encoder = ECGFounder(self.config.ecg_configs)
                    self.config.ecg_configs.num_filters = self.ecg_encoder.ecg_founder.filter_list
                    self._load_and_freeze(self.ecg_encoder, self.config.ecg_configs, "ecg_encoder", skip_loading=False)
                else:
                    raise NotImplementedError
            else:
                self.ecg_encoder = ResNet1D(self.config.ecg_configs)
                # ResNet1D already loads its own pretrained_weights internally, only apply freeze/finetune here
                self._load_and_freeze(self.ecg_encoder, self.config.ecg_configs, "ecg_encoder", skip_loading=True)
            if self.config.ecg_configs.get("use_ecg_fuser", True):  # default is true, old configs used this
                self.ecg_fuser = create_mlp(
                    input_dim=self.config.ecg_configs.num_filters[-1] * self.config.ecg_configs.get("max_ecgs_in_seq", 2),
                    layers_dim=self.config.ecg_configs.fuse_pair_hidden_sizes,
                    dropout_ratio=self.config.ecg_configs.dropout_ratio,
                )
                self.num_out_channels = self.config.ecg_configs.fuse_pair_hidden_sizes[-1]
            else:
                # times 2 for 2 ECGs
                self.num_out_channels = self.config.ecg_configs.num_filters[-1] * self.config.ecg_configs.get("max_ecgs_in_seq", 2),

        if self.config.hemo_configs.use_baseline_hemo:
            self.num_out_channels += self.config.hemo_configs.num_filters[-1]
            if self.config.hemo_configs.use_waveform:
                self.hemodynamics_encoder = ResNet1D(self.config.hemo_configs)
                # ResNet1D already loads its own pretrained_weights internally, only apply freeze/finetune here
                self._load_and_freeze(self.hemodynamics_encoder, self.config.hemo_configs, "hemodynamics_encoder", skip_loading=True)
            else:
                self.hemodynamics_encoder = create_mlp(
                    input_dim=len(self.config.hemo_configs.input_channels),
                    layers_dim=self.config.hemo_configs.num_filters,
                    dropout_ratio=self.config.hemo_configs.dropout_ratio
                )
                self._load_and_freeze(self.hemodynamics_encoder, self.config.hemo_configs, "hemodynamics_encoder", skip_loading=False)
        if self.config.time_configs.use_delta_t:
            self.num_out_channels += self.config.time_configs.hidden_sizes[-1]
            self.time_encoder = create_mlp(
                input_dim=1,
                layers_dim=self.config.time_configs.hidden_sizes,
                dropout_ratio=self.config.time_configs.dropout_ratio
            )
            self._load_and_freeze(self.time_encoder, self.config.time_configs, "time_encoder", skip_loading=False)
        if self.config.get("pretrained_weights"):
            self.load_pretrained_weights(self.config.pretrained_weights)
        if self.config.get("freeze_weights"):
            for param in self.parameters():
                param.requires_grad = False

    def load_pretrained_weights(self, file):
        checkpoint = torch.load(file)
        missing_keys, unexpected_keys = self.load_state_dict(checkpoint["model_state"], strict=False)
        if len(missing_keys) > 0:
            self.load_state_dict(checkpoint["model_state"], strict=False)
        key_start_idx = len(f"encoder.")
        weights = {k[key_start_idx:]: v for (k, v) in checkpoint["model_state"].items() if k[:key_start_idx] == "encoder."}
        self.load_state_dict(weights, strict=False)  # strict=False to load model trained with other inputs but don't need. TODO: why am I loading 3 times

    def _load_and_freeze(self, module, sub_config, default_pretrained_key, skip_loading):
        """Per-sub-encoder checkpoint loading and freezing, mirroring the top-level pretrained_weights/freeze_weights
        pattern: set pretrained_weights to load a checkpoint (to finetune, load it and leave freeze_weights false),
        and set freeze_weights: true to freeze the loaded weights. skip_loading is True for sub-encoders (e.g.
        ResNet1D) that already load their own config.pretrained_weights internally, to avoid loading it twice.
        """
        if not skip_loading:
            pretrained_weights = sub_config.get("pretrained_weights")
            if pretrained_weights:
                self._load_submodule_weights(module, pretrained_weights, sub_config.get("pretrained_key") or default_pretrained_key)
        if sub_config.get("freeze_weights"):
            for param in module.parameters():
                param.requires_grad = False

    def _load_submodule_weights(self, module, file, pretrained_key):
        checkpoint = torch.load(file)
        prefix = f"encoder.{pretrained_key}."
        weights = {k[len(prefix):]: v for k, v in checkpoint["model_state"].items() if k.startswith(prefix)}
        module.load_state_dict(weights, strict=False)

    def embed_ecgs(self, batch_dict):
        num_ecgs = self.config.ecg_configs.get("max_ecgs_in_seq", 2)
        ecg_embeddings = []
        for i in range(num_ecgs):
            waves_key, channel_key = f"waves_{i}", f"channel_names_{i}"
            if waves_key not in batch_dict or channel_key not in batch_dict:  # loop runs for the (number of ecgs) times
                ecg_embeddings.append(torch.zeros_like(ecg_embeddings[-1]))  # this should always work bc there's >= 2 ECGs
            else:
                waves = batch_dict[waves_key].cuda()
                batch_size, _, num_samples = waves.shape
                ecg_channel_data = []
                for channel in self.config.ecg_configs.input_channels:
                    ecg_batch_idxs, ecg_channel_idxs = np.where(batch_dict[channel_key] == channel)
                    ecg_channel_data.append(waves[ecg_batch_idxs, ecg_channel_idxs])
                ecg_channel_data = torch.stack(ecg_channel_data, dim=1)
                ecg_embeddings.append(self.ecg_encoder(ecg_channel_data))
        ecg_embeddings = torch.cat(ecg_embeddings, dim=1)
        return ecg_embeddings

    def fuse_ecgs(self, ecg_embeddings):
        if self.config.ecg_configs.get("use_ecg_fuser", True):  # default true, old configs used the fuser
            ecg_embeddings = self.ecg_fuser(ecg_embeddings)
        return ecg_embeddings

    def forward(self, batch_dict):
        encoder_representation = []

        if self.config.ecg_configs.get("use_ecg", True):
            ecg_embeddings = self.embed_ecgs(batch_dict)
            ecg_embeddings = self.fuse_ecgs(ecg_embeddings)
            encoder_representation.append(ecg_embeddings)

        if self.config.hemo_configs.use_baseline_hemo:
            # If the baseline hemo data consists of the full waveform:
            if all([ch_name in batch_dict["channel_names_0"] for ch_name in self.config.hemo_configs.input_channels]):
                waves = batch_dict[f"waves_0"].cuda()
                hemo_channel_data = []
                for channel in self.config.hemo_configs.input_channels:
                    hemo_batch_idxs, hemo_channel_idxs = np.where(batch_dict["channel_names_0"] == channel)
                    hemo_channel_data.append(waves[hemo_batch_idxs, hemo_channel_idxs])
                hemo_channel_data = torch.stack(hemo_channel_data, dim=1)
                if not self.config.hemo_configs.use_waveform:
                    if "rounded_mpap_0" in batch_dict:
                        hemo_channel_data = batch_dict["rounded_mpap_0"].float().view(-1, 1).cuda()
                    else:
                        hemo_channel_data = hemo_channel_data.mean(dim=-1)  # [B, num channels]
            else:  # only the mPAP is available in the dataset, directly use this as `hemo_channel_data`
                base_mpap_key = "rounded_mpap_0" if "rounded_mpap_0" in batch_dict else "mpap_0"
                hemo_channel_data = batch_dict[base_mpap_key].float().view(-1, 1).cuda()
            if self.config.hemo_configs.get("add_noise", False):
                noise_range = self.config.hemo_configs.noise_range[1] - self.config.hemo_configs.noise_range[0]
                noise = torch.rand_like(hemo_channel_data[:, :1]) * noise_range + self.config.hemo_configs.noise_range[0] # inclusive
                hemo_channel_data += noise
                batch_dict["baseline_mpap_noise"] = noise
            encoder_representation.append(self.hemodynamics_encoder(hemo_channel_data))  # [B, num_filters[-1]]

        if self.config.time_configs.use_delta_t:
            if self.config.time_configs.get("use_log_norm", False):
                delta_ts = batch_dict["log_delta_ts"]
            else:
                delta_ts = batch_dict["delta_ts"]
            if not isinstance(delta_ts, torch.Tensor):
                # Saliency script converts to tensor first and sets requires_grad. Don't want to create a copy in that case
                delta_ts = torch.tensor(delta_ts, dtype=torch.float)
            encoder_representation.append(self.time_encoder(delta_ts.float().view(-1, 1).cuda()))

        batch_dict["encoder_representation"] = torch.cat(encoder_representation, dim=1)

        return batch_dict
