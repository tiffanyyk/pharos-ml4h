"""Load a trained PHAROS multiclass model from a checkpoint + cfg file and run
one dummy forward pass. The dummy batch is built to match the data format
expected by the model.

Usage:
    python pharos_multiclass_example.py \
        --cfg_file pharos_config_demo.yaml \
        --ckpt_path pharos_multiclass.pth
"""

import argparse
import pickle
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from scipy import interpolate

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import cfg, cfg_from_yaml_file


def parse_args():
    parser = argparse.ArgumentParser(description="Run one dummy forward pass through a PHAROS regression model.")
    parser.add_argument("--cfg_file", type=str, default=str(REPO_ROOT / "demos/pharos_multiclass_demo/pharos_config_demo.yaml"),
                         help="path to the cfg yaml (MODEL.ENCODER / MODEL.DECODER configs)")
    parser.add_argument("--ckpt_path", type=str, default=str(REPO_ROOT / "demos/pharos_multiclass_demo/pharos_multiclass.pth"),
                         help="path to the checkpoint (either a raw state_dict, or a dict with a 'model_state' key)")
    parser.add_argument("--batch_size", type=int, default=12, help="batch size for the dummy input")
    return parser.parse_args()


def is_valid_ecg(ecg):
    """Given a single ECG, return False if it has voltages exceeding 5 mV or below -5 mV, or if there is no/poor signal.
    """
    if np.any(np.abs(ecg) > 5) or np.max(ecg) - np.min(ecg) < 0.1:
        return False
    return True


def build_dummy_batch(data_cfg, encoder_cfg, batch_size):
    """Build a dummy batch_dict matching what the forward pass of the model expects."""
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)

    batch_dict = {}

    # One batch of lead I ECGs in the shape of [batch size, num leads (= 1), num samples]
    ecg_cfg = encoder_cfg.ecg_configs
    if ecg_cfg.get("use_ecg", True):
        input_channels = ecg_cfg.input_channels
        seq_len = data_cfg.get("input_samples_per_seq", 4096)
        for i in range(2):  # 2 ecgs (baseline and followup)
            batch_dict[f"waves_{i}"] = torch.randn(batch_size, len(input_channels), seq_len)
            batch_dict[f"channel_names_{i}"] = np.array([input_channels for _ in range(batch_size)])

    # One batch of baseline mPAP that is z-score normalized according to our train set mean and std dev
    hemo_cfg = encoder_cfg.hemo_configs
    if hemo_cfg.use_baseline_hemo and not hemo_cfg.use_waveform:
        # normalized baseline mPAP value, one per sample
        batch_dict["rounded_mpap_0"] = (torch.randint(10, 40, (batch_size,)) - data_cfg.norm_params.pap[0]) / \
                                       data_cfg.norm_params.pap[1]

    # One batch of delta time gaps between the baseline and followup ECGs. Raw time gaps are in seconds.
    # They are log-normalized according to our train set mean and std dev.
    # Raw delta times should be non-negative.
    time_cfg = encoder_cfg.time_configs
    if time_cfg.use_delta_t:
        assert time_cfg.use_log_norm
        raw_delta_times = torch.rand(batch_size) * (365 * 24 * 60 * 60)  # e.g. random dummy delta times within a year
        raw_delta_times = torch.log(raw_delta_times + 1e-8)
        batch_dict["log_delta_ts"] = (raw_delta_times - data_cfg.norm_params.log_delta_time[0]) / \
                                     data_cfg.norm_params.log_delta_time[1]

    # These inputs are placeholders, we are using echo features only in training and not in external evaluation
    echo_cfg = encoder_cfg.get("echo_parameters_configs", {})
    if echo_cfg.get("use_echo_parameters"):
        num_features = echo_cfg.input_size // 2
        batch_dict["echo_features_1"] = np.zeros((batch_size, num_features))
        batch_dict["echo_missing_flag_1"] = np.ones((batch_size, num_features))

    return batch_dict


def build_sample_input(data_cfg, encoder_cfg):
    with open("../baseline_data_shared.pkl", "rb") as f:
        baseline_data = pickle.load(f)
    with open("../followup_data_shared.pkl", "rb") as f:
        followup_data = pickle.load(f)

    batch_size = 1
    num_leads = 1
    batch_dict = {}

    # Check ECG quality
    assert is_valid_ecg(baseline_data["lead_i"])
    assert is_valid_ecg(followup_data["lead_i"])

    # Resample them to have 4096 samples
    ecgs = []
    for data in [baseline_data, followup_data]:
        reqd_x = np.linspace(0, data_cfg.data_len_sec, data_cfg.input_samples_per_seq)
        data_x = np.linspace(0, data_cfg.data_len_sec, data["lead_i"].shape[0])
        wave_func = interpolate.interp1d(data_x, data["lead_i"], fill_value="extrapolate")
        ecgs.append(wave_func(reqd_x))

    # Model inputs
    batch_dict["waves_0"] = torch.tensor(ecgs[0]).float()
    batch_dict["channel_names_0"] = np.array(baseline_data["channel_names"])
    batch_dict["waves_1"] = torch.tensor(ecgs[1]).float()
    batch_dict["channel_names_1"] = np.array(followup_data["channel_names"])
    batch_dict["rounded_mpap_0"] = (torch.tensor(baseline_data["pap"]).mean().round() - data_cfg.norm_params.pap[0]) / \
                                    data_cfg.norm_params.pap[1]
    batch_dict["log_delta_ts"] = (torch.log(followup_data["timestamp"] - baseline_data["timestamp"] + 1e-8)
                                  - data_cfg.norm_params.log_delta_time[0]) / \
                                  data_cfg.norm_params.log_delta_time[1]

    # Ground truth label
    followup_mpap = (torch.tensor(followup_data["pap"]).mean().round() - data_cfg.norm_params.pap[0]) / \
                     data_cfg.norm_params.pap[1]
    batch_dict["delta_mpap"] = followup_mpap - batch_dict["rounded_mpap_0"]

    # Reshape for batch size
    for k, v in batch_dict.items():
        if "waves" in k:
            batch_dict[k] = v.reshape(batch_size, num_leads, -1)
        else:
            batch_dict[k] = v.reshape(batch_size, -1)

    # These inputs are placeholders, we are using echo features only in training and not in external evaluation
    echo_cfg = encoder_cfg.get("echo_parameters_configs", {})
    if echo_cfg.get("use_echo_parameters"):
        num_features = echo_cfg.input_size // 2
        batch_dict["echo_features_1"] = np.zeros((1, num_features))
        batch_dict["echo_missing_flag_1"] = np.ones((1, num_features))

    return batch_dict


def get_classification_labels_from_delta_mpap(delta_mpap, thresholds):
    labels = np.zeros_like(delta_mpap)
    for thresh in thresholds:
        labels += (delta_mpap >= thresh).astype(np.float32)
    return labels


def main():
    args = parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    model = torch.load(args.ckpt_path, weights_only=False)
    model.cuda()
    model.eval()

    batch_dict = build_dummy_batch(cfg.DATA_CONFIG, cfg.MODEL.ENCODER, args.batch_size)

    with torch.no_grad():
        output = model(batch_dict)
        output["scores"] = torch.nn.Softmax(dim=1)(batch_dict["classification_predictions"])

    norm_std = cfg.DATA_CONFIG.norm_params.pap[1]

    print(f"predicted_probabilities: {output['scores']}")

    batch_dict = build_sample_input(cfg.DATA_CONFIG, cfg.MODEL.ENCODER)
    with torch.no_grad():
        output = model(batch_dict)
        output["scores"] = torch.nn.Softmax(dim=1)(batch_dict["classification_predictions"])

    print(f"predicted probabilities: {output['scores']}")  # [[0.3230, 0.4702, 0.2068]]
    labels = get_classification_labels_from_delta_mpap(batch_dict['delta_mpap'].numpy() * norm_std, cfg.MODEL.DECODER.threshold)
    print(f"ground truth class index: {labels}")
    print(f"unnormalized ground truth delta_mpap: {batch_dict['delta_mpap'] * norm_std}")


if __name__ == "__main__":
    main()
