# Loss Masking Is Not Needed in Decoder-only Transformer for Discrete-token-based ASR

This repository contains the code for our ICASSP 2024
paper [LOSS MASKING IS NOT NEEDED IN DECODER-ONLY TRANSFORMER FOR DISCRETE-TOKEN-BASED ASR](https://arxiv.org/abs/2311.04534).

## Overview

Recently, unified speech-text models, such as SpeechGPT, VioLA, and AudioPaLM, have achieved remarkable performance on various speech tasks. These models discretize speech signals into tokens (speech discretization) and use a shared vocabulary for both text and speech tokens. Then they train a single decoder-only Transformer on a mixture of speech tasks. However, these models rely on the Loss Masking strategy for the ASR task, which ignores the dependency among speech tokens. 
In this paper, we propose to model speech tokens in an autoregressive way, similar to text. We find that applying the conventional cross-entropy loss on input speech tokens does not consistently improve the ASR performance over the Loss Masking approach. To address this issue, we propose a novel approach denoted Smoothed Label Distillation (SLD), which applies a KL divergence loss with smoothed labels on speech tokens. Our experiments show that SLD effectively models speech tokens and outperforms Loss Masking for decoder-only Transformers in ASR tasks with different speech discretization methods.
![](figure/sld.png)

## Installation

### Clone the repo

```shell
git clone https://github.com/alibaba-damo-academy/SpokenNLP.git
```

### Install Conda

```shell
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
conda create -n sld python=3.7
conda activate sld
```

### Install other packages

```shell
cd SpokenNLP/sld
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

## Training and test

We recommend run the script stage by stage to have an overview of our method.

```bash
bash run.sh
```

## Citation

If this project are helpful to your research, please cite:

```shell
@article{chen2023ditto,
  author       = {Qian Chen and
                  Wen Wang and
                  Qinglin Zhang and
                  Siqi Zheng and
                  Shiliang Zhang and
                  Chong Deng and
                  Yukun Ma and
                  Hai Yu and
                  Jiaqing Liu and
                  Chong Zhang},
  title        = {LOSS MASKING IS NOT NEEDED IN DECODER-ONLY TRANSFORMER FOR DISCRETE-TOKEN-BASED ASR},
  booktitle    = {ICASSP 2024},
  year         = {2024},
}
```
