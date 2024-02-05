# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

import fairseq
import soundfile as sf
import torch
import torch.nn.functional as F
import math

from feature_utils import get_path_iterator, dump_feature
from fairseq.data.audio.audio_utils import get_features_or_waveform
from WavLM import WavLM, WavLMConfig
from typing import Callable, Optional, Sequence, Tuple, Union
from torchaudio.transforms import Resample

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_wavlm_feature")


def _source_target_sample_rate(orig_freq: int, speed: float) -> Tuple[int, int]:
    source_sample_rate = int(speed * orig_freq)
    target_sample_rate = int(orig_freq)
    gcd = math.gcd(source_sample_rate, target_sample_rate)
    return source_sample_rate // gcd, target_sample_rate // gcd


class WavlmFeatureReader(object):
    def __init__(self, ckpt_path, layer, speed_factor, max_chunk=1600000):
        # load the pre-trained checkpoints
        checkpoint = torch.load(ckpt_path)
        cfg = WavLMConfig(checkpoint['cfg'])
        model = WavLM(cfg)
        model.load_state_dict(checkpoint['model'])

        self.model = model.eval().cuda()
        self.cfg = cfg
        self.layer = layer
        self.max_chunk = max_chunk
        self.speed_factor = speed_factor
        logger.info(f"TASK CONFIG:\n{self.cfg}")
        logger.info(f" max_chunk = {self.max_chunk}")

    def read_audio(self, path, ref_len=None):
        wav = get_features_or_waveform(path, need_waveform=True, use_sample_rate=16000)
        if wav.ndim == 2:
            wav = wav.mean(-1)
        if self.speed_factor != 1.0:
            source_sample_rate, target_sample_rate = _source_target_sample_rate(16000, self.speed_factor)
            resampler = Resample(orig_freq=source_sample_rate, new_freq=target_sample_rate)
            wav = torch.from_numpy(wav)
            wav = resampler(wav)
            wav = wav.numpy()
        assert wav.ndim == 1, wav.ndim
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, path, ref_len=None):
        x = self.read_audio(path, ref_len=ref_len)
        with torch.no_grad():
            x = torch.from_numpy(x).float().cuda()
            if self.cfg.normalize:
                x = F.layer_norm(x, x.shape)
            x = x.view(1, -1)

            feat = []
            for start in range(0, x.size(1), self.max_chunk):
                x_chunk = x[:, start: start + self.max_chunk]
                feat_chunk, _ = self.model.extract_features(
                    source=x_chunk,
                    padding_mask=None,
                    mask=False,
                    output_layer=self.layer,
                )
                feat.append(feat_chunk)
        return torch.cat(feat, 1).squeeze(0)


def main(tsv_dir, split, ckpt_path, layer, nshard, rank, feat_dir, speed_factor, max_chunk):
    reader = WavlmFeatureReader(ckpt_path, layer, speed_factor, max_chunk)
    generator, num = get_path_iterator(f"{tsv_dir}/{split}.tsv", nshard, rank)
    dump_feature(reader, generator, num, split, nshard, rank, feat_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_dir")
    parser.add_argument("split")
    parser.add_argument("ckpt_path")
    parser.add_argument("feat_dir")
    parser.add_argument("layer", type=int)
    parser.add_argument("nshard", type=int)
    parser.add_argument("rank", type=int)
    parser.add_argument("speed_factor", type=float, default=1.0)
    parser.add_argument("--max_chunk", type=int, default=1600000)
    args = parser.parse_args()
    logger.info(args)

    main(**vars(args))
