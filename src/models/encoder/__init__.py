from .contrastive_12to1_lead_ecg_encoder import Contrastive12To1LeadECGEncoder
from .ecg_founder import ECGFounder
from .ecg_pair_pa_deltat_encoder import ECGPairPADeltatEncoder
from .resnet1d import ResNet1D

__all__ = {
    "ResNet1D": ResNet1D,
    "ECGPairPADeltatEncoder": ECGPairPADeltatEncoder,
    "ECGFounder": ECGFounder,
    "Contrastive12To1LeadECGEncoder": Contrastive12To1LeadECGEncoder,
}


def build_encoder(config):
    model = __all__[config.NAME](
        config=config
    )
    if config.NAME == "Nothing":
        # Set the property according to configs, only used to initialize decoder
        model.num_out_channels = config.num_out_channels

    return model
