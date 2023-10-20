# Improving Long Document Topic Segmentation Models With Enhanced Coherence Modeling

The official repository of our EMNLP 2023 paper "**Improving Long Document Topic Segmentation Models With Enhanced Coherence Modeling**".

# Environment

```json
git clone git@github.com:alibaba-damo-academy/SpokenNLP.git
cd ./SpokenNLP/emnlp2023-topic_segmentation
conda create -n torch1.12.1 python=3.8
source activate torch1.12.1
pip install -r requirements.txt
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
In addition, download Punkt Tokenizer Model from [here](https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt.zip),
then `unzip punkt.zip` and move punkt folder to `path/to/your/anaconda3/envs/torch1.12.1/share/nltk_data/tokenizers`.
# Dataset

After downloading and decompressing the source data, including [WikiSection](https://github.com/sebastianarnold/WikiSection), [Elements](http://groups.csail.mit.edu/rbg/code/mallows/), [WIKI-727K and WIKI-50](https://github.com/koomri/text-segmentation),
you need to specify the source data path in `./config/config.ini`. 
Note that after decompressing WikiSection by running `unzip WikiSection-master.zip`, 
you need to further run `cd WikiSection-master && tar -xvzf wikisection_dataset_json.tar.gz` to get the data.

Then, you can run `bash run_process_data.sh` to process the data into the required unified format to `./data` folder.

# Pretrained Models

Downloading following pretrained models to `./pretrained_models`folder.

longformer_base: [https://huggingface.co/allenai/longformer-base-4096](https://huggingface.co/allenai/longformer-base-4096/)

bigbird_base: [https://huggingface.co/google/bigbird-roberta-base](https://huggingface.co/google/bigbird-roberta-base)

bert_base: [https://huggingface.co/bert-base-uncased](https://huggingface.co/bert-base-uncased/)

electra_base: [https://huggingface.co/google/electra-base-discriminator](https://huggingface.co/google/electra-base-discriminator)


# Training
Change some key parameters like `model_name` and`dataset` in `run_finetune.sh`.

Then run `bash run_finetune.sh`.

# Inference
Change some key parameters like `model_name` and `model_path` in `run_inference.sh`.

Then run `bash run_inference.sh`.


# Citation

If you find our code or our paper useful for your research, please **[★star]** this repo and **[cite]** our paper.

```
@article{improving,
  title={Improving Long Document Topic Segmentation Models With Enhanced Coherence Modeling},
  author={Hai Yu, Chong Deng, Qinglin Zhang, Jiaqing Liu, Qian Chen, Wen Wang},
  journal={arXiv preprint arXiv:2310.11772},
  year={2023}
}
```

# License

Licensed under the [Apache License 2.0](https://github.com/alibaba-damo-academy/SpokenNLP/blob/main/LICENSE). This project contains various third-party components under other open source licenses.
