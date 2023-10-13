
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="/mnt/workspace/workgroup/yuhai/paper_topic_seg/pretrained_models/longformer",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    sentence_pooler_type: str = field(
        default=None,
        metadata={"help": ""}
    )
    num_topic_labels: int = field(
        default=0,
        metadata={"help": "number of topic labels to predict in topic classification task"}
    )
    do_da_ts: bool = field(
        default=False,
        metadata={"help": "whether to compute topic segmentation loss on augmented example"}
    )
    do_cssl: bool = field(
        default=False,
        metadata={"help": "whether to compute cssl loss on original example"}
    )
    do_tssp: bool = field(
        default=False,
        metadata={"help": "whether to compute tssp loss on augmented example"}
    )
    ts_loss_weight: float = field(
        default=1.0,
        metadata={"help": "ts_loss_weight"}
    )
    ts_score_predictor: str = field(
        default="lt",
        metadata={"help": "how to get topic segmentation score from encodered sequence embedding. choices are ['lt', 'cos'], where lt means linear transform, cos means cosine similarity"}
    )
    ts_score_predictor_cos_temp: float = field(
        default=1,
        metadata={"help": "work when ts_score_predictor is cos"}
    )
    focal_loss_gamma: float = field(
        default=0.0,
        metadata={"help": "focal_loss_gamma for ts"}
    )
    weight_label_zero: float = field(
        default=0.5,
        metadata={"help": "for weighted loss. weight_label_zero is the weight of B-EOP, 1 - weight_label_zero is the weight of O"}
    )
    cl_loss_weight: float = field(
        default=0.0,
        metadata={"help": "contrastive learning loss weight."}
    )
    cl_temp: float = field(
        default=1,
        metadata={"help": "temp of cosine similarity, to scale softmax(similarity) to increase the difference between similarity scores"}
    )
    cl_anchor_level: str = field(
        default="eop_matrix",
        metadata={"help": "choices are [eop_matrix, eot_matrix, eop_list, eot_list]. matrix means extract all features, and compute similarity matrix. list means extract anchor features, then compute repective cl loss"}
    )
    cl_positive_k: int = field(
        default=1,
        metadata={"help": "work when cl_anchor_level is eot. get k nearest eop features in the same topic with anchor eot feature, if number is smaller than k, then repeat randomly for the convenience"}
    )
    cl_negative_k: int = field(
        default=1,
        metadata={"help": "work when cl_anchor_level is eot. get k nearest eop features in next topics"}
    )
    tssp_loss_weight: float = field(
        default=0.0,
        metadata={"help": "loss weight for sentence ordering prediction task on augmented data"}
    )
    tssp_ablation:str = field(
        default="none",
        metadata={"help": "choices are ['none', 'wo_intra_topic', 'wo_inter_topic', 'sso]"}
    )
    num_gpu: int = field(
        default=1,
        metadata={"help": "number of gpu. for computing eval_steps"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="topic_segment", metadata={"help": "The name of the task (ner, pos...)."})
    metric_name: Optional[str] = field(
        default="./metrics/seqeval.py", metadata={"help": "The name of the metric to use (via the datasets library)."}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_cache_dir: Optional[str] = field(
        default="../..//cached_data", metadata={"help": "cache tokenizered data"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    text_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of text to input in the file (a csv or JSON file)."}
    )
    label_column_name: Optional[str] = field(
        default=None, metadata={"help": "The column name of label to input in the file (a csv or JSON file)."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=4096,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. If set, sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    eval_cnt: int = field(
        default=50,
        metadata={"help": "how many total times to do validation"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to put the label for one word on all tokens of generated by that word or just on the "
                "one (in which case the other tokens will have a padding index)."
            )
        },
    )
    return_entity_level_metrics: bool = field(
        default=True,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        self.task_name = self.task_name.lower()


@dataclass
class CustomArguments:
    test_data_name: str = field(
        default=None,
        metadata={"help": "to discriminate different test set."}
    )
    threshold: float = field(
        default=0.5,
        metadata={"help": "only segmentation point with prediction score >= threshold can be keeped as segmentation point predict result"}
    )
    topk: int = field(
        default=None,
        metadata={"help": "only segmentation point with topk prediction score can be keeped as segmentation point predict result"}
    )
    topk_with_threshold: bool = field(
        default=None,
        metadata={"help": "just choose topk scores which are bigger than threshold"}
    )
    f1_at_k: int = field(
        default=None,
        metadata={"help": "soft f1 which will take predictioin boundary as hitting if \
         there are ground truths within k sentences to the left and right of the prediction boundary"}
    ) 
    