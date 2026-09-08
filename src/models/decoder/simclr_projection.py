import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.utils.create_networks import create_mlp


class SimCLRProjection(nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        in_channels1, in_channels2 = in_channels
        self.config = config
        out_features = self.config.out_features if self.config.out_features is not None else in_channels
        if self.config.get("use_nonlinear_map"):
            self.projection1 = create_mlp(input_dim=in_channels1,
                                          layers_dim=self.config.nonlinear_layers,
                                          dropout_ratio=self.config.dropout_ratio)
            self.projection2 = create_mlp(input_dim=in_channels2,
                                          layers_dim=self.config.nonlinear_layers,
                                          dropout_ratio=self.config.dropout_ratio)
        else:
            self.projection1 = nn.Linear(in_features=in_channels1, out_features=out_features, bias=False)
            self.projection2 = nn.Linear(in_features=in_channels2, out_features=out_features, bias=False)
        self.temperature = self.config.get("temperature", 0.1)

    @staticmethod
    def l2_normalize(representation):
        """input tensor: batch, dim"""
        return F.normalize(representation, p=2, dim=-1)

    def forward(self, batch_dict):
        repr_keys = [key for key in batch_dict.keys() if key.endswith("representation")]
        assert len(repr_keys) == 2, "SimCLR projection expects exactly 2 embeddings to contrast"

        embed1 = self.l2_normalize(self.projection1(batch_dict[repr_keys[0]]))
        embed2 = self.l2_normalize(self.projection2(batch_dict[repr_keys[1]]))

        batch_dict["projected_embed1"] = embed1
        batch_dict["projected_embed2"] = embed2

        return batch_dict

    def get_loss(self, batch_dict, tb_pre_tag=""):
        # NT-Xent / SimCLR loss, reference: https://arxiv.org/abs/2002.05709
        embed1, embed2 = batch_dict["projected_embed1"], batch_dict["projected_embed2"]
        batch_size = embed1.shape[0]
        device = embed1.device

        embeddings = torch.cat([embed1, embed2], dim=0)  # [2N, D], rows [0, N) are view 1, [N, 2N) are view 2
        similarity = torch.matmul(embeddings, embeddings.t()) / self.temperature  # [2N, 2N]

        self_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        similarity = similarity.masked_fill(self_mask, float("-inf"))

        # for each anchor row, the positive is the same sample's other view
        positive_idxs = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(0, batch_size, device=device),
        ])

        loss = F.cross_entropy(similarity, positive_idxs)

        predictions = similarity.argmax(dim=-1)
        accuracy = (predictions == positive_idxs).float().mean().item()

        tb_dict, disp_dict = {}, {}
        tb_dict[f"{tb_pre_tag}simclr_loss"] = loss.item()
        disp_dict[f"{tb_pre_tag}simclr_loss"] = loss.item()
        tb_dict[f"{tb_pre_tag}simclr_accuracy"] = accuracy
        disp_dict[f"{tb_pre_tag}simclr_accuracy"] = accuracy

        batch_dict["loss"] = loss.item()

        if "additional_stats" not in batch_dict:
            batch_dict["additional_stats"] = {}
        batch_dict["additional_stats"].update(tb_dict)

        return loss, tb_dict, disp_dict
