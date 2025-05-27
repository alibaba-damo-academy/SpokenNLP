import os
import json
import datasets


_CITATION = """"""
_DESCRIPTION = """"""

task = "topic_segmentation"
version = datasets.Version("1.1.1")
desc = "avlecture"


class TopicSegmentationConfig(datasets.BuilderConfig):
    """BuilderConfig for DS."""

    def __init__(self, **kwargs):
        """BuilderConfig for DS.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(TopicSegmentationConfig, self).__init__(**kwargs)


class AvlMultiModalityForTopicSegmentation(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        TopicSegmentationConfig(name="AvlMultiModalityForTopicSegmentation", version=version, description=desc),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    "example_id": datasets.Value("int32"),
                    "lecture": datasets.Value("string"),
                    "sentences": datasets.Sequence(datasets.Value("string")),
                    "labels": datasets.Sequence(datasets.Value("string")),
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
            # Homepage of the dataset for documentation
            homepage="",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": os.path.join(dl_manager.manual_dir, "train.jsonl"),
            "dev": os.path.join(dl_manager.manual_dir, "dev.jsonl"),
            "test": os.path.join(dl_manager.manual_dir, "test.jsonl")
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        """Yields examples."""
        print(task)
        label_map = {"1":"B-EOP", "0":"O", 1:"B-EOP", 0:"O"}       # B-EOP means end sentence of topic
        with open(filepath, "r") as f:
            for index, line in enumerate(f.readlines()):
                example = json.loads(line.strip())
                sentences = example["text"]
                labels = [label_map[v] if v in label_map else -100 for v in example["labels"]]      # label 1 in data file means end sentence of topic
                yield index, {
                    "example_id": index,
                    "lecture": example["example_id"].split("@@")[1],
                    "sentences": sentences,
                    "labels": labels,
                }