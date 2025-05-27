## Introduction 

This is the official repository of our ACL Findings 2025 paper **Multimodal Fusion and Coherence Modeling for Video Topic Segmentation**.  


## Environment


```angular2html
git clone git@github.com:alibaba-damo-academy/SpokenNLP.git
cd ./SpokenNLP/acl2025-multimodal_video_topic_segmentation
conda create -n torch1.12.1 python=3.8
source activate torch1.12.1
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```

## Dataset
Description of data format:
```json
{
  "example_id": "id of example",
  "text": "clip level asr transcript",
  "vid_duration": "duration of each clip, unit is second",
  "stet": "start time and end time of each clip, unit is second.",
  "labels": "binary label of whether the clip is last boundary of the topic it belong to. 1 means yes.",
  "topic_end_seconds": "end time of each topic, unit is second",
  "large_topic_labels": "binary label of coarse topic.",
  "large_topic_labels": "end time of coarse topic.",
  "type": "video type, such as blackboard, ppt, other.",
  "url": "video url, used to identify the source of CLVTS videos."
}
```
The original AVLecture dataset can be accessed [here](https://cvit.iiit.ac.in/research/projects/cvit-projects/avlectures). For our work, we divided the original test set into training, development, and test subsets, following a 7:1:2 ratio.
## Pretrained Models
Downloading following pretrained models to `./pretrained_models` folder.  
English Text Encoder: [longformer_base](https://huggingface.co/allenai/longformer-base-4096)  
Chinese Text Encoder: [longformer_zh](https://huggingface.co/ValkyriaLenneth/longformer_zh)

## Training
There are two types of training: one is to use only **Text** features, and the other is to use **Multimodal** features.

- Text topic segmentation
```
bash run_finetune_text.sh
```

- Multimodal topic segmentation

Due to the ethical considerations, we only public the video URL, the transcribed text, and the manually annotated topic segmentation boundaries. If you notice any infringement, please don't hesitate to reach out to us at your earliest convenience. We are committed to handling such matters promptly and appreciate your cooperation.  

Therefore, if you want to run multimodal topic segmentation, please download the video to local environment first, and then extract visual features and audio features according to the descriptions in the paper.

Then, you need re-implement pytorch version MoE layers based on [the TensorFlow implementation](https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py) to run model with MoE components.

After that, change the multimodal feature path in `run_finetune_multimodal.sh` and run `bash run_finetune_multimodal.sh`.

## Citation
If the video topic segmentation dataset and methods in this project are helpful to your research, please cite:
```
@article{yu2024multimodal,
  title={Multimodal Fusion and Coherence Modeling for Video Topic Segmentation},
  author={Yu, Hai and Deng, Chong and Zhang, Qinglin and Liu, Jiaqing and Chen, Qian and Wang, Wen},
  journal={arXiv preprint arXiv:2408.00365},
  year={2024}
}
```


## License

The CLVTS dataset, developed by Alibaba Group, is intended strictly for academic and non-commercial purposes and is distributed under the Apache License (Version 2.0).