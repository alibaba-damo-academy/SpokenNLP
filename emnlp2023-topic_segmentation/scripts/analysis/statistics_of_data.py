
import os
import json
import pandas as pd

from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt

from .tokenizer import BasicTokenizer

bTokenizer = BasicTokenizer()
tokenize_func = bTokenizer.tokenize


def data_statistics(in_file):
    # "篇章字符长度", "句子数", "话题数", "句子长度", "话题长度", "话题句子数"
    describe_df_output_path = os.path.join(os.path.dirname(in_file), "{}_describe_df.csv".format(os.path.basename(in_file)))
    
    doc_length = []
    doc_sent_cnts = []
    doc_paragraph_cnts = []
    doc_topic_cnts = []
    sent_length = []
    topic_length = []
    topic_sent_cnts = []

    paragraph_length = []
    paragraph_sent_cnts = []
    topic_paragraph_cnts = []
    
    with open(in_file, "r") as f:
        for i, line in enumerate(f.readlines()):
            example = json.loads(line.strip())
            sentences, labels = example["sentences"], example["labels"]

            doc_sent_cnts.append(len(sentences))
            doc_paragraph_cnts.append(len([v for v in labels if v != -100]))
            doc_topic_cnts.append(len([v for v in labels if v == 1]))

            d_len = 0
            topic_sents = []
            paragraph_sents = []
            topic_paragraphs = []
            for sent, label in zip(sentences, labels):
                sent_length.append(len(tokenize_func(sent)))
                topic_sents.append(sent)
                if label == 1:
                    topic_sent_cnts.append(len(topic_sents))
                    t_len = len(tokenize_func("".join(topic_sents)))
                    topic_length.append(t_len)
                    d_len += t_len
                    topic_sents = []
                
                paragraph_sents.append(sent)
                if label != -100:
                    paragraph_length.append(len(tokenize_func("".join(paragraph_sents))))
                    paragraph_sent_cnts.append(len(paragraph_sents))
                    topic_paragraphs.append(paragraph_sents)
                    if label == 1:
                        topic_paragraph_cnts.append(len(topic_paragraphs))
                        topic_paragraphs = []
                    paragraph_sents = []


            doc_length.append(d_len)
    
    describe_df = pd.DataFrame(doc_length, columns=["篇章字符长度"]).describe()
    describe_df = pd.concat([describe_df, pd.DataFrame(doc_sent_cnts, columns=["句子数"]).describe()] ,axis=1)
    describe_df = pd.concat([describe_df, pd.DataFrame(doc_topic_cnts, columns=["话题数"]).describe()] ,axis=1)
    describe_df = pd.concat([describe_df, pd.DataFrame(sent_length, columns=["句子长度"]).describe()] ,axis=1)
    describe_df = pd.concat([describe_df, pd.DataFrame(topic_length, columns=["主题长度"]).describe()] ,axis=1)
    describe_df = pd.concat([describe_df, pd.DataFrame(topic_sent_cnts, columns=["主题句子数"]).describe()] ,axis=1)

    describe_df.to_csv(describe_df_output_path, encoding="utf-8-sig")
    

if __name__ == "__main__":
    in_file = "path/to/your/data/file"
    data_statistics(in_file)
    