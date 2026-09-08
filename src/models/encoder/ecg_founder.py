import os

import torch
from huggingface_hub import hf_hub_download

from src.models.encoder.ecg_founder_utils.net1d import Net1D

NUM_LEADS_TO_CHECKPOINT_FILENAME = {
    1: "1_lead_ECGFounder.pth",
    12: "12_lead_ECGFounder.pth",
}


class ECGFounder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        num_leads = self.config.get("num_leads", 1)
        checkpoint_filename = NUM_LEADS_TO_CHECKPOINT_FILENAME[num_leads]

        saved_model_path = f"/home/tyau/pa-informed-ecg/src/models/encoder/ecg_founder_utils/{checkpoint_filename}"
        if not os.path.exists(saved_model_path):
            local_dir = os.path.dirname(saved_model_path)
            os.makedirs(local_dir, exist_ok=True)
            hf_hub_download(repo_id="PKUDigitalHealth/ECGFounder", local_dir=local_dir, filename=checkpoint_filename)

        self.ecg_founder = Net1D(
            in_channels=num_leads,
            base_filters=64,  # 32 64
            ratio=1,
            filter_list=[64, 160, 160, 400, 400, 1024, 1024],
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
            kernel_size=16,
            stride=2,
            groups_width=16,
            verbose=False,
            use_bn=False,
            use_do=False,
            n_classes=1,
            return_features_only=True,  # added this arg to skip the dense layer
        )
        checkpoint = torch.load(saved_model_path, weights_only=False)
        state_dict = checkpoint["state_dict"]
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('dense.')}
        self.ecg_founder.load_state_dict(state_dict, strict=False)
        del self.ecg_founder.dense  # will put dense layer outside the encoder

        if self.config.get("freeze_weights", True):
            for param in self.ecg_founder.parameters():
                param.requires_grad = False

        self.num_out_channels = self.ecg_founder.filter_list[-1]

    def forward(self, x):
        x = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-8)
        return self.ecg_founder(x)


if __name__ == "__main__":
    ECGFounder({})
