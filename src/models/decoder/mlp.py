import torch.nn

from src.models.utils.create_networks import create_mlp


class MLP(torch.nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        self.in_channels = in_channels
        self.config = config
        self.layers = create_mlp(in_channels, config.nonlinear_layers, config.dropout_ratio)
        if self.config.get("pretrained_weights"):
            self.load_pretrained_weights(self.config.pretrained_weights)

    def load_pretrained_weights(self, file):
        checkpoint = torch.load(file)
        key_start_idx = len(f"decoder.")
        weights = {k[key_start_idx:]: v for (k, v) in checkpoint["model_state"].items() if k[:key_start_idx] == "decoder."}
        self.load_state_dict(weights, strict=False)

        # reinitialize the last n layers, n based on cfg
        if self.config.get("randomize_last_layer"):
            self.layers[-1].reset_parameters()

        if self.config.get("freeze_weights"):
            for param in self.parameters():
                param.requires_grad = False
            # TODO: if unfreezing more than 1 layer, would need to update this to check specifically for Linear (vs dropout, relu)
            if self.config.get("unfreeze_last_layer"):
                for param in self.layers[-1].parameters():
                    param.requires_grad = True

    def forward(self, batch_dict):
        keys = [key for key in batch_dict if "representation" in key]
        if len(keys) > 1:
            raise ValueError(f"More than one representation in {keys}")
        outputs = self.layers(batch_dict[keys[0]].cuda())
        for prefix in ["regression", "classification"]:
            if prefix in self.config.task:
                batch_dict[f"{prefix}_predictions"] = outputs
        return batch_dict

