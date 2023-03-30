# Meeting Action Item Detection with Regularized Context Modeling

- [**Meeting Action Item Detection with Regularized Context Modeling**](https://arxiv.org/abs/2303.16763)
- *Jiaqing Liu, Chong Deng, Qinglin Zhang, Qian Chen, Wen Wang*

## Introduction

Meetings are an increasingly important way to collaborate. 
With the support of automatic speech recognition, 
the meetings could be transcribed into transcripts. 
In meeting transcripts, action items are crucial for managing post-meeting to-do tasks, 
which usually are summarized laboriously. 
Action item detection is aimed to detect meeting content associated with action items automatically. 

For action item detection, the relevant datasets are scarce and small-scale. 
Thus, we construct and release the [AMC-A](https://www.modelscope.cn/datasets/modelscope/Alimeeting4MUG/summary) corpus, 
which is the first Chinese meeting corpus with action item annotations.

Based on the corpus, 
we propose the **Context-Drop** approach to utilize both local and global contexts by contrastive learning, 
and achieve better performance and robustness in the action item detection task. 
In addition, we explore the **Lightweight Model Ensemble** method to exploit different pre-trained models

The [paper](https://arxiv.org/abs/2303.16763) has been accepted by ICASSP 2023.

## Installation
### Clone the repo

```shell
git clone https://github.com/alibaba-damo-academy/SpokenNLP.git
```

### Install Conda

```shell
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh Miniconda3-latest-Linux-x86_64.sh
conda create -n action python=3.7
conda activate action
```

### Install other packages
```shell
pip install -r requirements.txt
```

## Corpus
### AMI

The [AMI meeting corpus](https://groups.inf.ed.ac.uk/ami/corpus/index.shtml) has played an essential role in various meeting-related research. 
It contains 171 meeting transcripts and various types of annotations. 
Among them, there are 145 scenario-based meetings and 26 naturally occurring meetings. 

The AMI meeting corpus is a common dataset for benchmarking action item detection systems.
Although there are no direct annotations for action items for this corpus, 
indirect annotations can be generated based on annotations of the summary. 
Following previous works, we consider dialogue acts linked to the action-related abstractive summary as positive samples and otherwise negative samples. 
In this way, we obtain 101 annotated meetings with 381 action items.
In addition, we apply the official scenario-only dataset partitioning.

The AMI corpus can be processed according to the following steps:
1. Download the AMI corpus and third-party annotations (i.e., [ami_public_manual_1.6.2.zip](http://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip)). 
2. Unzip ami_public_manual_1.6.2.zip to ```data/AMI/ami_public_manual_1.6.2/``` path.
3. Run ```data_script/ami_process.py``` and get the dataset in the ```data/AMI/dataset/``` path.

### AMC-A

We construct and make available a Mandarin meeting corpus, the [AliMeeting-Action Corpus](https://www.modelscope.cn/datasets/modelscope/Alimeeting4MUG/summary) (AMC-A), with action item annotations. 
We extend 224 meetings previously published in [M2MET](https://arxiv.org/abs/2110.07393) with additional 200 meetings. 
Each meeting session consists of a 15-minute to 30-minute discussion by 2-4 participants covering certain topics from a diverse set, biased towards work meetings in various industries.
All 424 meeting recordings are manually transcribed with punctuation inserted. 
Semantic units ended with a manually labeled period, question mark, and exclamation are treated as **sentences** for action item annotations and modeling.

We formulate action item detection as a binary classification task and conduct sentence-level action item annotation, i
.e., sentences containing action item information (task description, time frame, owner) as positive samples (labeled as 1) and otherwise negative samples (labeled as 0). 
As found in previous research and our experience, annotations of action items have high subjectivity and low consistency, e.g., only a Kappa coefficient of 0.36 on the ICSI corpus.

To ease the task, we provide detailed annotation guidelines with sufficient examples.  
To reduce the annotation cost, we first select candidate sentences containing both temporal expressions (e.g., "tomorrow") and action-related verbs (e.g., "finish"), and highlight them in different colors.
Candidate sentences are then annotated by three annotators independently. 
When annotating, candidate sentences are presented with their context so that annotators can easily exploit context information. 

With these quality control methods, the average Kappa coefficient between pairs of annotators is **0.47**. 
For inconsistent labels from three annotators, an expert reviews the majority voting results and decides on the final labels. 
The following table shows that AMC-A has much more meeting sessions, total utterances, and total action items than the AMI meeting corpus and comparable avg. action items per meeting.
To the best of our knowledge, AMC-A is so far the first Chinese meeting corpus and the largest meeting corpus in any language labeled for action item detection.

Statistic of Corpus          |    AMC-A    | AMI
-----------------------------| :---------: | :------:
Total # Meeting              |   **424**   | 101
Total # Utterances           | **306,846** | 80,298
Total # Action               |   **1506**  | 381
Kappa Coefficient            |    0.47     | /
Avg. # Action per Meeting    |    3.55     | 3.77
Std. # Action per Meeting    |    3.97     | 1.95

## Action Item Detection


We formulate action item detection as a binary classification task.
Given an utterance `X` with its context `C`, the model predicts the label `y`, i.e., whether `X` contains action items or not. 
Context-Drop explores local and global contexts together with regularization. Lightweight Model Ensemble is an efficient approach for improving performance using different pre-trained models while preserving inference latency.

Before running action item detection, you should process the dataset to some directory `$DATA_DIR` as described above.
And download the pre-trained checkpoint such as [BERT](https://github.com/google-research/bert/) or [StructBERT](https://github.com/alibaba/AliceMind/tree/main/StructBERT) and unzip it to some directory `$PRETRAIN_DIR`.


### Training the classifier

You can `bash go_train.sh` to train the classifier for the action item detection task.
This example code is as follows:

```shell
CUDA_VISIBLE_DEVICES=0 python run_classifier.py \
  --task_name=meet  \
  --vocab_file=${PRETRAIN_DIR}/vocab.txt \
  --bert_config_file=${PRETRAIN_DIR}/bert_config.json \
  --init_checkpoint=${PRETRAIN_DIR}/${PRETRAIN_CKPT} \
  --do_train=True \
  --do_eval=True \
  --do_predict=True \
  --do_lower_case=True \
  --classifier_input=${CLASS_INPUT} \
  --classifier_model=${CLASS_MODEL} \
  --context_type=${CONTEXT_TYPE} \
  --dev_context_type=${DEV_CONTEXT_TYPE} \
  --test_context_type=${TEST_CONTEXT_TYPE} \
  --context_width=${CONTEXT_WIDTH} \
  --noisy_type=${NOISY_TYPE} \
  --threshold=${THRESHOLD} \
  --do_label_smoothing=${LOSS_SMOOTH} \
  --loss_type=${LOSS_TYPE} \
  --dropout_rate=${DROPOUT_RATE} \
  --kl_alpha=${KL_ALPHA} \
  --drop_type=${DROP_TYPE} \
  --data_dir=${DATA_DIR} \
  --record_dir=record/${DATA_NAME} \
  --train_file=${DATA_DIR}/${TRAIN_FILE} \
  --dev_file=${DATA_DIR}/${DEV_FILE} \
  --test_type="file" \
  --test_path=${DATA_DIR}/${TEST_FILE} \
  --predict_dir=${OUTPUT_DIR}/predict/file_test/ \
  --model_log_file=model.log.txt \
  --output_dir=${OUTPUT_DIR}/ \
  --best_model_dir=${BEST_DIR}/ \
  --do_export=False \
  --do_frozen=False \
  --export_dir=export/ \
  --max_seq_length=${MAX_LENGTH} \
  --train_batch_size=${BATCH_SIZE} \
  --eval_batch_size=32 \
  --learning_rate=${LEARNING_RATE} \
  --warmup_proportion=0.1 \
  --save_checkpoints_steps=100 \
  --train_summary_steps=10 \
  --num_train=${NUM_EPOCH} \
  --num_train_type="epoch"
```

You can modify some experimental setting, such as:
- `$CLASS_INPUT`: the input of classifier, including `cls` (default), `sep`, `token_avg`, `token_max`. 
- `$CONTEXT_TYPE`: the contextual input type, including `sentence`, `sentence+left`, `sentence+right`, `sentence+left+right`, `left+sentence+right`, `sentence+left+right+global`, etc.
- `$CONTEXT_WIDTH`: the width of contextual input.
- `$DROP_TYPE`: the contrastive learning method type, including `r-drop`, `context-drop-fix`, `context-drop-dynamic`.

Please see the [paper](https://arxiv.org/abs/2303.16763) for more details.

Note: You can run `python night_listener.py` along with `bash go_train.sh`, 
so that you can save the best models based on F1 during the evaluation period.


### Prediction from classifier

Once you have trained your classifier, you can use `bash go_predict.sh` to predict.
You can specify the test file to be predicted. 
Output includes predicted probabilities for each sample and the overall metrics.

Note: Please keep the parameters consistent with `go_train.sh`.

### Repeated experiments

You can use `bash repeat_models.sh` to run repeated experiments, and specify the hyper-parameters such as `$LEARNING_RATE` and `$NUM_EPOCH`.
Then you can use `python average_performance.py` to compute the average performance.

## Experiment


We use both the AMI meeting corpus and our Chinese meeting corpus (AMC-A).
Considering the sparsity of positive samples, we report positive F1 as the evaluation metric.


Base Model    |             Input               |          Method        | AMC-A F1 | AMI F1
--------------|---------------------------------| :---------------------:| :------:| :------:
BERT          | sentence                        | None                   |  64.76  |  38.18
StructBERT    | sentence                        | None                   |  67.84  |  38.67
StructBERT    | sentence                        | R-Drop                 |  68.77  |  39.26
StructBERT    | sentence + local context        | None                   |  68.50  |  41.03
StructBERT    | sentence + local context        | R-Drop                 |  68.79  |  42.72
StructBERT    | sentence + local context        | Context-Drop (fixed)   |  69.15  |  43.12
StructBERT    | sentence + local context        | Context-Drop (dynamic) |  69.53  |  42.05
StructBERT    | sentence + global context       | None                   |  67.99  |  35.82
StructBERT    | sentence + global context       | Context-Drop (dynamic) |  70.48  |  41.25
StructBERT    | sentence + local&global context | None                   |  69.09  |  41.31
StructBERT    | sentence + local&global context | Context-Drop (dynamic) |  70.82  |  41.50

Note: The AMC-A test dataset in the paper is different from the test dataset of the [ICASSP 2023 MUG Challenge Track5 Action Item Detection (AID)](https://modelscope.cn/competition/17/summary). 
The test dataset in the paper contains 64 meetings. 
On this basis, we add 100 new meetings and split them into test1 of phase1 and test2 of phase2 containing 82 meetings respectively. 
Here, we attach the performance on test1 and test2 of the action item detection track.


Base Model    |             Input               |          Method        | AMC-A Test1 F1 | AMC-A Test2 F1
--------------|---------------------------------| :---------------------:| :-------------:| :------------:
BERT          | sentence                        | None                   |      65.10     |  60.27
StructBERT    | sentence                        | None                   |      67.09     |  62.49
StructBERT    | sentence                        | R-Drop                 |      68.36     |  64.54
StructBERT    | sentence + local context        | None                   |      66.56     |  63.33
StructBERT    | sentence + local context        | R-Drop                 |      68.65     |  64.19
StructBERT    | sentence + local context        | Context-Drop (fixed)   |      67.85     |  63.57
StructBERT    | sentence + local context        | Context-Drop (dynamic) |      68.73     |  64.25
StructBERT    | sentence + global context       | None                   |      67.85     |  62.50
StructBERT    | sentence + global context       | Context-Drop (dynamic) |      68.98     |  65.04
StructBERT    | sentence + local&global context | None                   |      67.70     |  64.23
StructBERT    | sentence + local&global context | Context-Drop (dynamic) |      68.60     |  65.31

## Citing
If the AMC-A corpus and methods in this project are helpful to your research, please cite:

```shell
@inproceedings{liu2023action,
  title={Meeting Action Item Detection with Regularized Context Modeling},
  author={Liu, Jiaqing and Deng, Chong and Zhang, Qinglin and Chen, Qian and Wang, Wen},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2023},
  organization={IEEE}
}
```

## Acknowledgement
We borrowed a lot of code from [Google BERT](https://github.com/google-research/bert/).

## License
This action-item-detection project is developed by Alibaba and based on the google-research BERT project
Code is distributed under the Apache License (Version 2.0)
This product contains various third-party components under other open source licenses. 
See the NOTICE file for more information.




