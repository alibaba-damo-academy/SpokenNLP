
from .ma_encoder import MergeAttentionEncoder
from .ca_encoder import CoAttentionEncoder
from .ma_moe_encoder import MergeAttentionMoEEncoder
from .ca_moe_encoder import CoAttentionMoEEncoder


def get_cross_encoder(config):
    cross_encoder_type = getattr(config, "cross_encoder_type", "ma")
    
    if config.cross_encoder_type == "ma":
        return MergeAttentionEncoder(config)
    elif config.cross_encoder_type == "ca":
        return CoAttentionEncoder(config)
    elif config.cross_encoder_type == "ma_moe":
        return MergeAttentionMoEEncoder(config)
    elif config.cross_encoder_type == "ca_moe":
        return CoAttentionMoEEncoder(config)
    else:
        raise ValueError("not support cross_encoder_type: {}".format(config.cross_encoder_type))