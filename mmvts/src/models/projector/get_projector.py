
from .linear_projector import LinearProjector
from .transformer_projector import TransformerProjector


def get_projector(config):
    proj_type = getattr(config, "proj_type", "linear")
    if proj_type == "linear":
        return LinearProjector(config)
    else:
        return TransformerProjector(config)
    