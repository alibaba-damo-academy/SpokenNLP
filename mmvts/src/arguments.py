
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default="",
        metadata={"help": "Path to finetuned multi-modality model"}
    )
    init_model: bool = field(
        default=True,
        metadata={"help": "if True, then init mm model else load from model_name_or_path"}
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
    weight_label_zero: float = field(
        default=0.5,
        metadata={"help": "for weighted loss. weight_label_zero is the weight of B-EOP, 1 - weight_label_zero is the weight of O"}
    )
    num_gpu: int = field(
        default=1,
        metadata={"help": "number of gpu. for computing eval_steps"}
    )
    # fuse modality features
    use_raw_shot: bool = field(
        default=False,
        metadata={"help": "whether to use raw shot or extracted shot representations"}
    )
    max_vis_seq_length: int = field(
        default=300,
        metadata={"help": "max_vis_seq_length"}
    )
    use_vis2d: bool = field(
        default=True,
        metadata={"help": "whether to use vis 2d feature"}
    )
    use_vis3d: bool = field(
        default=False,
        metadata={"help": "whether to use vis 3d feature"}
    )
    use_vis_ocr: bool = field(
        default=False,
        metadata={"help": "whether to use ocr feature"}
    )
    text_encoder_name_or_path: str=field(
        default="longformer_base",
        metadata={"help": "name or path of text encoder"}
    )
    vis2d_encoder_name: str=field(
        default="clip_vit_b_32",
        metadata={"help": "name of vis 2d encoder"}
    )
    vis3d_encoder_name: str=field(
        default="resnext101",
        metadata={"help": "name of vis 3d encoder"}
    )
    vis_ocr_encoder_name: str=field(
        default="sbert",
        metadata={"help": "name of vis ocr encoder"}
    )
    audio_encoder_name: str=field(
        default="whisper_small"
    )
    hidden_size_vis2d: int = field(
        default=2048,
        metadata={"help": "hidden_size of vis 2d feature"}
    )
    hidden_size_vis3d: int = field(
        default=2048,
        metadata={"help": "hidden_size of vis 3d feature"}
    )
    hidden_size_vis_ocr: int = field(
        default=768,
        metadata={"help": "hidden_size of ocr feature"}
    )
    freeze_text_encoder: bool = field(
        default=False,
        metadata={"help": "whether to freeze text encoder"}
    )
    freeze_vis2d_encoder: bool = field(
        default=False,
        metadata={"help": "whether to freeze vis encoder"}
    )
    hidden_size_audio: int=field(
        default=768,
    )
    cross_encoder_type: str = field(
        default="ca",
        metadata={"help": "ca means self_attention and cross_attention, moe means cat feature, compute self-attention, moe ffn"}
    )
    num_cross_encoder_layers: int = field(
        default=0,
    )
    num_cross_encoder_heads: int = field(
        default=12,
    )
    cross_encoder_lr: float = field(
        default=5e-5,
    )
    cross_moe_share_in_layers: bool = field(
        default=False,
    )
    cross_moe_num_experts: int = field(
        default=4,
    )
    cross_moe_top_k: int = field(
        default=2,
    )
    cross_moe_input_size: int = field(
        default=768,
    )
    cross_moe_output_size: int = field(
        default=768,
    )
    cross_moe_hidden_size: int = field(
        default=3072,
    )
    cross_moe_lw: float = field(
        default=1.0,
    )
    cross_moe_residual: bool = field(
        default=True,
    )
    out_modal_prob: bool = field(
        default=False,
    )
    fuse_type: str = field(
        default="cat",
        metadata={"help": "how to fuse vis and text feature, choices are cat, add"}
    )
    hidden_dropout_prob: float=field(
        default=0.1,
        metadata={"help": "overwrite hidden_dropout_prob"}
    )
    ts_lw: float = field(
        default=1.0
    )
    # contrastive learning
    cl_temp: float = field(
        default=0.1,
    )
    do_modality_cl: bool = field(
        default=False,
        metadata={"help": ""}
    )
    do_align_av: bool = field(
        default=False,
        metadata={"help": ""}
    )
    align_av_weight: float = field(
        default=0.33,
    )
    do_align_at: bool = field(
        default=False,
        metadata={"help": ""}
    )
    align_at_weight: float = field(
        default=0.33,
    )
    do_align_tv: bool = field(
        default=False,
        metadata={"help": ""}
    )
    align_tv_weight: float = field(
        default=0.33,
    )
    align_before_fuse: bool = field(
        default=False,
    )
    modality_cl_lw: float = field(
        default=0.0,
    )
    do_topic_mm_cl: bool = field(
        default=False
    )
    topic_mm_cl_lw: float = field(
        default=0.0
    )
    topic_cl_type: str = field(
        default="list",
        metadata={"help": "choices are list and matrix."}
    )
    topic_cl_choice: str = field(
        default="near",
    )
    topic_cl_fct: str = field(
        default="simcse",
        metadata={"help": "choices are ce, simcse. ce means cross entropy, simcse means deno and no"}
    )
    topic_mm_cl_pos_k: int = field(
        default=1,
        metadata={"help": "work when topic_cl_type is list"}
    )
    topic_mm_cl_neg_k: int = field(
        default=3,
        metadata={"help": "work when topic_cl_type is list"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    language_type: str = field(
        default="en",
        metadata={"help": "choices are en, cn"}
    )
    task_name: Optional[str] = field(default="topic_segment", metadata={"help": "The name of the task (ner, pos...)."})
    metric_name: Optional[str] = field(
        default="./metrics/seqeval.py", metadata={"help": "The name of the metric to use (via the datasets library)."}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_cache_dir: Optional[str] = field(
        default="../../cached_data", metadata={"help": "cache tokenizered data"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    dataset_root_dir: str = field(
        default="../data/avlecture",
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a csv or JSON file)."}
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
        default=3,
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
    vis_frame_dir: str=field(
        default="/mnt/workspace/workgroup/yuhai/lecture_video_seg/data/avlecture/frames_1fps",
    )
    vis2d_feature_cache_dir: str=field(
        default="/mnt/workspace/workgroup/yuhai/lecture_video_seg/data/avlecture/cws/features/visual_2d_resnet152-max",
    )
    vis3d_feature_cache_dir: str=field(
        default="/mnt/workspace/workgroup/yuhai/lecture_video_seg/data/avlecture/cws/features/visual_3d_resnext101-max",
    )
    vis_ocr_feature_cache_dir: str=field(
        default="/mnt/workspace/workgroup/yuhai/lecture_video_seg/data/avlecture/cws/features/visual_ocr_sbert/ocr",
    )
    vis_embedding_pooling: str=field(
        default="max",
        metadata={"help": "pooling vis features in one clip to one hidden_size"}
    )
    audio_feature_cache_dir: str=field(
        default="",
        metadata={"help": "path of audio feature"}
    )
    audio_info_file: str=field(
        default="",
        metadata={"help": "path of audio second info"}
    )
    audio_clip_token_num: int=field(
        default=1500,
        metadata={"help": "token num of clip audio feature"}
    )
    audio_sample_hz: int=field(
        default=50,
        metadata={"help": "30s * 50 -> 1500, so audio_sample_hz is 50"}
    )
    infer_train_data: bool = field(
        default=False,
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
    pretrain_task: str = field(
        default="ts",
        metadata={"help": "ts or align"}
    )
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
    