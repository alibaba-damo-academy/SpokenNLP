
import os
import json
from pathlib2 import Path
from tqdm import tqdm


def abridge_model_name(model_name_or_path):
    model_name = os.path.basename(model_name_or_path)
    if "longformer" in model_name_or_path:
        model_name = "lf"
    elif "bigbird" in model_name_or_path:
        model_name = "bb"
    elif "bert" in model_name_or_path:
        model_name = "bert"
    elif "electra" in model_name_or_path:
        model_name = "ele"
    else:
        raise ValueError("not supported model_name")
    return model_name


def convert_res_format(file_path, custom_args):
    out_path = os.path.join(os.path.dirname(file_path), os.path.basename(file_path).split(".json")[0] + "_str_metric.txt")
    with open(file_path, "r") as f:
        res = json.load(f)
    
    threshold_example_level_precision = res["threshold_%s_example_level_precision" % (custom_args.threshold)]
    threshold_example_level_recall = res["threshold_%s_example_level_recall" % (custom_args.threshold)]
    threshold_example_level_f1 = res["threshold_%s_example_level_f1" % (custom_args.threshold)]
    threshold_example_level_pk = res["threshold_%s_example_level_pk" % (custom_args.threshold)]
    threshold_example_level_wd = res["threshold_%s_example_level_wd" % (custom_args.threshold)]
    
    threshold_res_str = "threshold_%s_example_level_metric\n" % (custom_args.threshold) + \
        " / ".join(["%.2f" % (float(v) * 100) for v in [
            threshold_example_level_precision,
            threshold_example_level_recall,
            threshold_example_level_f1,
            threshold_example_level_pk,
            threshold_example_level_wd,
            ]])
    
    with open(out_path, "w") as f:
        f.write("p / r / f / pk / wd\n")
        f.write(threshold_res_str + "\n\n")

    print("p / r / f / pk / wd\n")
    print(threshold_res_str + "\n\n")
