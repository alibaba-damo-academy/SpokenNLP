
import os
import json
import glob

from pathlib2 import Path
from tqdm import tqdm


def get_hidden_size_vis(config):
    hidden_size_vis = 0
    if config.use_vis2d:
        hidden_size_vis += config.hidden_size_vis2d
    if config.use_vis3d:
        hidden_size_vis += config.hidden_size_vis3d
    if config.use_vis_ocr:
        hidden_size_vis += config.hidden_size_vis_ocr
    return hidden_size_vis


def get_in_predictor_hidden_size(config):
    in_predictor_hidden_size = config.hidden_size
    if config.fuse_type == "cat":
        in_predictor_hidden_size = config.hidden_size * 3
    elif config.fuse_type in ["cat_a_t", "cat_a_v", "cat_t_v"]:
        in_predictor_hidden_size = config.hidden_size * 2
    elif config.fuse_type in ["text_only", "vis_only", "audio_only", "cat_seq", "add", "mean", "max", "gate", "gate_v2"]:
        in_predictor_hidden_size = config.hidden_size
    else:
        raise ValueError("not supported fuse_type: {}".format(config.fuse_type))
    return in_predictor_hidden_size


def get_cross_encoder_kv_hidden_size(config):
    if config.fuse_type == "cat":
        # cat another two modality in hidden dim, then get K V to do QKV
        ce_kv_hidden_size = config.hidden_size * 2
    else:
        ce_kv_hidden_size = config.hidden_size
    return ce_kv_hidden_size


def update_config(config, model_args, data_args,):
    config.language_type = data_args.language_type
    
    config.init_model = model_args.init_model
    config.ts_lw = model_args.ts_lw
    
    config.text_encoder_name_or_path = model_args.text_encoder_name_or_path
    config.vis2d_encoder_name = model_args.vis2d_encoder_name
    config.vis3d_encoder_name = model_args.vis3d_encoder_name
    config.vis_ocr_encoder_name = model_args.vis_ocr_encoder_name
    config.audio_encoder_name = model_args.audio_encoder_name
    
    config.cache_dir = model_args.cache_dir
    config.model_revision = model_args.model_revision
    
    config.use_raw_shot = model_args.use_raw_shot
    config.max_vis_seq_length = model_args.max_vis_seq_length

    config.hidden_size_vis2d = model_args.hidden_size_vis2d
    config.hidden_size_vis3d = model_args.hidden_size_vis3d
    config.hidden_size_vis_ocr = model_args.hidden_size_vis_ocr
    config.hidden_size_audio = model_args.hidden_size_audio
    config.use_vis2d = model_args.use_vis2d
    config.use_vis3d = model_args.use_vis3d
    config.use_vis_ocr = model_args.use_vis_ocr
    config.freeze_text_encoder = model_args.freeze_text_encoder
    config.freeze_vis2d_encoder = model_args.freeze_vis2d_encoder
    
    config.cross_encoder_type = model_args.cross_encoder_type
    config.num_cross_encoder_layers = model_args.num_cross_encoder_layers
    config.num_cross_encoder_heads = model_args.num_cross_encoder_heads
    config.cross_encoder_lr = model_args.cross_encoder_lr
    config.ce_kv_hidden_size = get_cross_encoder_kv_hidden_size(config)
    config.cross_moe_num_experts = model_args.cross_moe_num_experts
    config.cross_moe_input_size = model_args.cross_moe_input_size
    config.cross_moe_output_size = model_args.cross_moe_output_size
    config.cross_moe_hidden_size = model_args.cross_moe_hidden_size
    config.cross_moe_top_k = model_args.cross_moe_top_k
    config.cross_moe_lw = model_args.cross_moe_lw
    config.cross_moe_residual = model_args.cross_moe_residual

    config.hidden_dropout_prob = model_args.hidden_dropout_prob

    config.fuse_type = model_args.fuse_type
    
    dataset_name = os.path.basename(data_args.dataset_name)
    data_dir = os.path.join(data_args.dataset_root_dir, dataset_name)
    config.data_dir = data_dir
    config.vis2d_feature_cache_dir = data_args.vis2d_feature_cache_dir
    config.vis3d_feature_cache_dir = data_args.vis3d_feature_cache_dir
    config.vis_ocr_feature_cache_dir = data_args.vis_ocr_feature_cache_dir
    config.audio_feature_cache_dir = data_args.audio_feature_cache_dir

    config.do_modality_cl = model_args.do_modality_cl
    config.align_before_fuse = model_args.align_before_fuse
    config.modality_cl_lw = model_args.modality_cl_lw
        
    config.do_align_av = model_args.do_align_av
    config.align_av_weight = model_args.align_av_weight
    config.do_align_at = model_args.do_align_at
    config.align_at_weight = model_args.align_at_weight
    config.do_align_tv = model_args.do_align_tv
    config.align_tv_weight = model_args.align_tv_weight

    config.do_topic_mm_cl = model_args.do_topic_mm_cl
    config.topic_mm_cl_lw = model_args.topic_mm_cl_lw
    config.topic_mm_cl_pos_k = model_args.topic_mm_cl_pos_k
    config.topic_mm_cl_neg_k = model_args.topic_mm_cl_neg_k
    
    config.topic_cl_type = model_args.topic_cl_type
    config.topic_cl_choice = model_args.topic_cl_choice
    config.topic_cl_fct = model_args.topic_cl_fct
    
    config.vis_frame_dir = data_args.vis_frame_dir
    config.train_file = data_args.train_file
    config.validation_file = data_args.validation_file
    config.test_file = data_args.test_file
    config.file_dict = {
        "train": os.path.join(data_dir, "train.jsonl"),
        "dev": os.path.join(data_dir, "dev.jsonl"),
        "test": os.path.join(data_dir, "test.jsonl"),
    }
        
    config.hidden_size_vis = get_hidden_size_vis(config)
    config.in_predictor_hidden_size = get_in_predictor_hidden_size(config)

    config.out_modal_prob = model_args.out_modal_prob
    
    print("config: ", config)
    return config


def get_example_cached_feature_paths(folder_path, lecture):
    pattern = os.path.join(folder_path, lecture + '*')
    feature_files = glob.glob(pattern)
    feature_files.sort()
    # print(len(feature_files))
    res = {}
    for feature_file_path in feature_files:
        feature_file_name = os.path.basename(feature_file_path)
        feature_clip_index = int(feature_file_name.split("-")[-1].split(".npy")[0])
        res[feature_clip_index] = feature_file_path
    # print(res)
    return res   


def get_audio_length_dict(audio_info_file):
    res = {}        # lecture -> clip index -> audio second
    with open(audio_info_file, "r") as f:
        for line in f.readlines():
            _, second, wav_path, feature_path = line.strip().split(",")
            clip_name = os.path.basename(feature_path).split(".npy")[0]
            feature = "-".join(clip_name.split("-")[:-1])
            clip_index = int(clip_name.split("-")[-1])
            if feature not in res:
                res[feature] = {}
            res[feature][clip_index] = second
    return res


def abridge_model_name(text_encoder_name_or_path, vis2d_encoder_name_or_path):
    text_encoder_name = os.path.basename(text_encoder_name_or_path)
    vis2d_encoder_name = os.path.basename(vis2d_encoder_name_or_path)

    if "Erlangshen-Longformer-110M" == text_encoder_name:
        text_encoder_name = "t_lf_els"
    elif "longformer_zh" == text_encoder_name:
        text_encoder_name = "t_lf_zh"
    elif "bert" in text_encoder_name:
        text_encoder_name = "t_bert"
    elif "longformer" in text_encoder_name:
        text_encoder_name = "t_lf"
    else:
        raise ValueError("not supported text_encoder_name: {}".format(text_encoder_name))

    return text_encoder_name

def abridge_dataset_name(dataset_name):
    if dataset_name == "avlecture":
        dataset_name = "avl"
    return dataset_name
    

def convert_res_format(file_path, custom_args):
    out_path = os.path.join(os.path.dirname(file_path), os.path.basename(file_path).split(".json")[0] + "_str_metric.txt")
    with open(file_path, "r") as f:
        res = json.load(f)
    
    threshold_example_level_precision = res["threshold_%s_example_level_precision" % (custom_args.threshold)]
    threshold_example_level_recall = res["threshold_%s_example_level_recall" % (custom_args.threshold)]
    threshold_example_level_f1 = res["threshold_%s_example_level_f1" % (custom_args.threshold)]
    threshold_example_level_pk = res["threshold_%s_example_level_pk" % (custom_args.threshold)]
    threshold_example_level_wd = res["threshold_%s_example_level_wd" % (custom_args.threshold)]
    threshold_example_level_avg_pred_cnt = res["threshold_%s_example_level_avg_pred_cnt" % (custom_args.threshold)]
    threshold_example_level_avg_true_cnt = res["threshold_%s_example_level_avg_true_cnt" % (custom_args.threshold)]
    
    threshold_res_str = "threshold_%s_example_level_metric\n" % (custom_args.threshold) + \
        " / ".join(["%.2f" % (float(v) * 100) for v in [
            threshold_example_level_precision,
            threshold_example_level_recall,
            threshold_example_level_f1,
            # threshold_example_level_pk,
            # threshold_example_level_wd,
            ]] + [str(threshold_example_level_avg_pred_cnt), str(threshold_example_level_avg_true_cnt)])
    
    topk_res_str = None
    if custom_args.topk is not None:
        topk_example_level_precision = res["topk_%s_example_level_precision" % (custom_args.topk)]
        topk_example_level_recall = res["topk_%s_example_level_recall" % (custom_args.topk)]
        topk_example_level_f1 = res["topk_%s_example_level_f1" % (custom_args.topk)]
        topk_example_level_pk = res["topk_%s_example_level_pk" % (custom_args.topk)]
        topk_example_level_wd = res["topk_%s_example_level_wd" % (custom_args.topk)]
        topk_example_level_avg_pred_cnt = res["topk_%s_example_level_avg_pred_cnt" % (custom_args.topk)]
        topk_example_level_avg_true_cnt = res["topk_%s_example_level_avg_true_cnt" % (custom_args.topk)]

        topk_res_str = "topk_%s_example_level_metric\n" % (custom_args.topk) + \
                    " / ".join(["%.2f" % (float(v) * 100) for v in [
                        topk_example_level_precision,
                        topk_example_level_recall,
                        topk_example_level_f1,
                        # topk_example_level_pk,
                        # topk_example_level_wd,
                    ]] + [str(topk_example_level_avg_pred_cnt), str(topk_example_level_avg_true_cnt)])
    
    with open(out_path, "w") as f:
        f.write("p / r / f / pk / wd / avg_pred_cnt / avg_true_cnt\n")
        f.write(threshold_res_str + "\n\n")
        if topk_res_str is not None:
            f.write("top{}\n".format(custom_args.topk))
            f.write(topk_res_str + "\n")

    print("p / r / f / pk / wd / avg_pred_cnt / avg_true_cnt\n")
    print(threshold_res_str + "\n\n")
    if topk_res_str is not None:
        print(topk_res_str)


def max_clip_cnt(data_file):
    res = -1
    res_example_id = ""
    with open(data_file, "r") as f:
        for line in f.readlines():
            example = json.loads(line)
            text = example["text"]
            if len(text) >= res:
                res = len(text)
                res_example_id = example["example_id"]
    print(res, res_example_id)
    return res


def check_vis_features():
    data_file = "/mnt/workspace/workgroup/yuhai/lecture_video_seg/data/avlecture/avl_text_clip_level_0110/all.jsonl"
    folder_path = "/mnt/workspace/workgroup/yuhai/lecture_video_seg/data/avlecture/cws/features/visual_2d_resnet152"
    # folder_path = "/mnt/workspace/workgroup/yuhai/lecture_video_seg/data/avlecture/cws/features/visual_ocr_sbert/ocr"
    lectures = []
    data = {}
    with open(data_file, "r") as f:
        for line in f.readlines():
            example = json.loads(line)
            example_ids = example["example_id"]
            lecture = example_ids.split("@@")[1]
            text = example["text"]
            data[lecture] = len(text)
            lectures.append(lecture)
    # print(lectures)
    # print(len(lectures))

    miss_fea = 0
    for lecture in lectures:
        # if lecture != "MIT8_701F20_01_300k":
        #     continue
        pattern = os.path.join(folder_path, lecture + '*')
        feature_files = glob.glob(pattern)
        feature_files.sort()
        # if len(feature_files) == 0:
        #     no_fea += 1
        #     print(lecture, feature_files)

        if len(feature_files) != data[lecture]:
            print(lecture, len(feature_files),  data[lecture])
            miss_fea += 1
    print(miss_fea)


if __name__ == "__main__":
    # folder_path = "/mnt/workspace/workgroup/yuhai/lecture_video_seg/data/avlecture/cws/features/visual_2d_resnet152-max"
    # example_id = "mit049@@MIT8_701F20_01_300k"
    # lecture = example_id.split("@@")[1]

    # folder_path = "/mnt/workspace/workgroup/yuhai/lecture_video_seg/data/visual_related/features/2d-cn_clip_vit_b_16-max-k3/"
    # lecture = "2024军队文职公共科目最全网课-马克思主义理论命题新趋势"

    folder_path = "/mnt/workspace/workgroup/yuhai/lecture_video_seg/data/cn_course_wo_labels_0523/features/3d-resnext101-max-k3/"
    lecture = "上海大学公开课：材料力学###No_35###第17课薄壁杆件扭转"
    lecture = "武汉大学公开课：名家（上）###No_56###武汉大学公开课：“性伪相分”的人文"
    res = get_example_vis_feature_paths(folder_path, lecture)
    # print(res)

    # data_file = "/mnt/workspace/workgroup/yuhai/lecture_video_seg/data/avlecture/avl_text_clip_level_0110/all.jsonl"
    # max_clip_cnt(data_file)

    # check_vis_features()
