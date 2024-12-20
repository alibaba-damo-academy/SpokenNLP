# SWAB Dataset

SWAB (**S**poken2**W**ritten of **A**SR transcripts **B**enchmark) is a benchmark for **Contextualized Spoken-to-Written conversion (CoS2W)** task. SWAB contains 60 *document-level* transcripts with manual annotations and auxiliary information, covering meeting, podcast, and lecture domains in both Chinese and English languages. More details about SWAB can be found in our [paper](https://arxiv.org/abs/2408.09688), which is accepted by AAAI 2025.

- [**Recording for Eyes, Not Echoing to Ears: Contextualized Spoken-to-Written Conversion of ASR Transcripts**](https://arxiv.org/abs/2408.09688)
- *Jiaqing Liu, Chong Deng, Qinglin Zhang, Shilin Zhou, Qian Chen, Hai Yu, Wen Wang*

## Introduction

Automatic Speech Recognition (ASR) transcripts exhibit recognition errors and various spoken language phenomena such as disfluencies, ungrammatical sentences, and incomplete sentences, hence suffering from poor readability. To improve readability, we propose a **Contextualized Spoken-to-Written conversion (CoS2W)** task to address ASR and grammar errors and also transfer the *informal* text into the *formal* style with content preserved, utilizing contexts and auxiliary information.

To promote research in this field, we construct and make available the *document-level* **Spoken2Written of ASR transcripts Benchmark (SWAB)** dataset with manual annotations, covering meeting, podcast, and lecture domains in both Chinese and English languages.
There are 60 transcripts with auxiliary information, with each subcategory comprising 10 documents. 
Please refer to the [paper](https://arxiv.org/abs/2408.09688) for more details.

The Chinese meetings in the SWAB are sourced from the training set of the [AliMeeting](https://www.modelscope.cn/datasets/modelscope/AliMeeting/files) dataset, 
and the English meetings come from the publicly available [AMI](https://groups.inf.ed.ac.uk/ami/corpus/) meeting dataset. 
The Chinese and English podcast and speech data are sourced from [YouTube](https://www.youtube.com/) and the [Tingwu](https://tingwu.aliyun.com/discover) website. 
For our collected podcasts and lectures, we only provide ASR transcripts after rigorous text anonymization processes and our annotations, to ensure transparency regarding the data sources and their usage while maintaining anonymity.

If you notice any infringement, please don't hesitate to reach out to us at your earliest convenience. We are committed to handling such matters promptly and appreciate your cooperation.

## Data Acquisition

In the swab_example.json file, we have shown results for only 6 documents, with one document per scenario. 
If you require the complete SWAB data of 60 documents, please contact us via the provided email address (mingzhai.ljq@alibaba-inc.com)
and specify the purpose of the data. We will provide the full dataset accordingly, ensuring ethical use of the dataset.


## Field Description

The meanings of the fields in the provided JSON file of the SWAB dataset are as follows:

```json
{
  "meeting_key": "title of the document",
  "sentences": [
    {
      "id": "sentence ID, starting from 1",
      "s": "sentence content, i.e., ASR recognition result",
      "speaker": "speaker ID",
      "s_gt": "ground truth of sentence content, i.e., human annotation result with manual ASR error corrections",
      "start_time": "start time of the sentence",
      "end_time": "end time of the sentence"
    },
    {}
  ],
  "paragraph_segment_ids": [
    {
      "id": "sentence ID at the end of the paragraph",
      "target": "human annotation result for the CoS2W task corresponding to the paragraph"
    },
    {}
  ],
  "description": "auxiliary Information",
  "language": "language, Chinese/English",
  "dataset_domain": "domain, including Chinese/English meeting/podcast/lecture",
  "link": "audio/video link"
}
```

## Data Statistics

The following table presents the statistics of the SWAB. 
It can be seen that both Chinese and English data contain sufficient paragraphs, providing ample support for the evaluation. 
Additionally, we have collected relevant auxiliary information. In particular, the podcast domain has a wealth of long text information.

| Domain          | # Words  | # Sentences | # Paragraphs | # Turns | # Speakers | # Auxiliary |
|-----------------|----------|:-----------:|:------------:|:-------:|:----------:|:-----------:|
| Chinese Podcast | 21610.10 |   683.10    |    202.70    | 191.70  |    2.90    |   2051.30   |
| Chinese Meeting | 9926.10  |   769.10    |    324.80    | 314.40  |    2.40    |   135.70    |
| Chinese Lecture | 7677.10  |   216.00    |    39.70     |  1.00   |    1.00    |    73.90    |
| English Podcast | 10077.40 |   709.00    |    232.00    | 100.00  |    3.30    |   192.70    |
| English Meeting | 3860.50  |   324.40    |    131.60    | 144.20  |    4.00    |   212.60    |
| English Lecture | 4182.00  |   245.90    |    47.40     |  1.00   |    1.00    |    91.00    |
| SWAB            | 9555.53  |   419.25    |    163.03    | 125.38  |    2.43    |   459.53    |


## Citation
If the SWAB dataset and methods in this project are helpful to your research, please cite:

```
@inproceedings{liu2024recording,
  title={Recording for Eyes, Not Echoing to Ears: Contextualized Spoken-to-Written Conversion of ASR Transcripts},
  author={Liu, Jiaqing and Deng, Chong and Zhang, Qinglin and Chen, Qian and Yu, Hai and Wang, Wen},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2025}
}

```

## License
The SWAB dataset, developed by Alibaba Group, 
is intended strictly for academic and non-commercial purposes 
and is distributed under the Apache License (Version 2.0).

