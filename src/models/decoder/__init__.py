from .mlp import MLP
from .simclr_projection import SimCLRProjection

__all__ = {
    "MLP": MLP,
    "SimCLRProjection": SimCLRProjection,
}


def build_decoder(in_channels, config):
    model = __all__[config.NAME](
        in_channels=in_channels,
        config=config
    )

    return model
