from transformers import BertTokenizer
import torch
from torch import nn, Tensor
import pandas as pd
import numpy as np
import glob
# from modeling_bertabs import BertAbsConfig, BertAbs, build_predictor
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torch.nn import functional as F
from transformers import BertTokenizer, BertModel
from collections import deque
import random


class RawDocTokenize():
    '''
    to tokenize the article and summary text into bert token id

    usage:
        rdt = RawDocTokenize(raw_article, raw_summary)
        tokenized_output = rdt.get_tokenized_output()

    input:
        raw_article, raw_summary: string of unprocessed text

    output of rdt.get_tokenized_output():
        dictionary with keys: ['src', 'tgt', 'segs', 'clss']
            src: bert token ids of article
            tgt: bert token ids of summary
            segs: sentence embeddings [000000111111000111] mark sentences
            clss: position of cls token (sentence begin token in src)
    '''

    def __init__(self, raw_article: str, raw_summary: str,
                 article_min_sentence_length: int = 6,
                 summary_min_sentence_length: int = 5,
                 article_max_token_length: int = None,
                 summary_max_token_length: int = None) -> object:
        '''
        raw_article, raw_summary: string of unprocessed text

        '''

        self.raw_article = raw_article
        self.raw_summary = raw_summary
        self.article_min_sentence_length = article_min_sentence_length
        self.summary_min_sentence_length = summary_min_sentence_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower=True)

        # max_token_length is only used when padding = True. For Bert work
        if article_max_token_length is not None:
            self.article_max_token_length = article_max_token_length
        else:
            self.article_max_token_length = 512  # article_max_token_length

        if summary_max_token_length is not None:
            self.summary_max_token_length = summary_max_token_length
        else:
            self.summary_max_token_length = 512  # summary_max_token_length

    def add_missing_period(self, line):
        END_TOKENS = [".", "!", "?", "...", "'", "`", '"', u"\u2019", u"\u2019", ")"]
        if line.startswith("@highlight"):
            return line
        if line[-1] in END_TOKENS and len(line):
            return line
        return line + "."

    def raw_text_process(self):
        article = self.raw_article
        article = article.replace("'", ' " ')
        article = [s.strip() + '.' for s in article.split('.')]
        article = [self.add_missing_period(line) for line in article if len(line) > 0]
        # remove sentences that are too short
        article = [s for s in article if s not in ['..', '.'] and len(s.split(' ')) >= self.article_min_sentence_length]

        summary = self.raw_summary
        summary = summary.replace("'", ' " ')
        summary = [s.strip() for s in summary.split('.')]
        # summary = [s.strip() + '.' for s in summary.split('.')]
        # summary = [self.add_missing_period(line) for line in summary if len(line) > 0]
        # # remove sentences that are too short
        # summary = [s for s in summary if s not in ['..', '.'] and len(s.split(' ')) >= self.summary_min_sentence_length]

        # print('from raw text process:')
        # print('article list broken = ', article[:13])
        # print('summary list broken = ', summary[:4])

        self.article = article
        self.summary = summary
        return article, summary

    def tokenize_article(self, padding: bool):
        '''
        add CLS and SEP token at sentence boundary
        '''
        #         min_sentence_length = self.article_min_sentence_length
        #         article = [item for item in self.article if len(item.split(' ')) >= min_sentence_length]

        encoding = self.tokenizer(self.article)
        src = [item for sublist in encoding['input_ids']
               for item in sublist]
        segs = []
        for i in range(len(encoding['token_type_ids'])):
            new_sub_token_type_ids = [segi + (i % 2) for segi in encoding['token_type_ids'][i]]
            segs += new_sub_token_type_ids

        cls_tokenid = self.tokenizer.vocab['[CLS]']

        # pad the remaining of the sentences if len[src] < 512
        if padding:
            max_token_len = self.article_max_token_length
            if len(src) > max_token_len:
                src = src[:max_token_len]
                segs = segs[:max_token_len]
            else:
                src += [-1] * (max_token_len - len(src))
                segs += [-1] * (max_token_len - len(src))

        clss = [i for i, tokenid in enumerate(src) if tokenid == cls_tokenid]  # position of CLS token (101)

        self.src = src
        self.segs = segs
        self.clss = clss
        return src, segs, clss

    def tokenize_summary(self, padding: bool):
        #         min_sentence_length = self.summary_mn_sentence_length
        #         summary = [item for item in self.summary if len(item.split(' ')) >= min_sentence_length]
        tgt_encoding = self.tokenizer(self.summary)
        decoder_symbols = {
            "BOS": self.tokenizer.vocab["[unused0]"],  # 1
            "EOS": self.tokenizer.vocab["[unused1]"],  # 2
            "TRG_SENT_SPLIT": self.tokenizer.vocab["[unused2]"], # 3
            "PAD": self.tokenizer.vocab["[PAD]"]  # 0
        }
        gt_summary_split = self.summary
        tgt = []
        for i, sent in enumerate(gt_summary_split):
            # sent = sent.replace("'", ' " ')
            #     print(sent)
            sent_token = self.tokenizer.encode(sent)
            if i == 0:
                sent_token[0] = 1  # unused 0
            if i > 0:
                sent_token = sent_token[1:]  # remove [CLS] at first
            # change all [sep] to unknown2
            sent_token[-1] = 3
            if i == len(gt_summary_split)-1:
                sent_token[-1] = 2

            tgt += sent_token

        # tgt = [item for sublist in tgt_encoding['input_ids'] for item in sublist]
        # for i, tokenid in enumerate(tgt):
        #     if tokenid == 101:  # CLS:
        #         tgt[i] = decoder_symbols['BOS']
        #     elif tokenid == 102:
        #         tgt[i] = decoder_symbols['TRG_SENT_SPLIT']
        #
        # tgt[-1] = decoder_symbols['EOS']

        # pad the remaining of the sentences if len[src] < 512
        if padding:
            max_token_len = self.summary_max_token_length
            if len(tgt) > max_token_len:
                tgt = tgt[:max_token_len]
            else:
                tgt += [-1] * (max_token_len - len(tgt))

        self.tgt = tgt
        return tgt

    def get_tokenized_output(self, padding: bool = False):
        key_names = ['src', 'tgt', 'segs', 'clss', 'src_txt', 'tgt_txt']
        article, summary = self.raw_text_process()
        _ = self.tokenize_article(padding)
        _ = self.tokenize_summary(padding)
        return {'src': self.src,
                'tgt': self.tgt,
                'segs': self.segs,
                'clss': self.clss}


if __name__ == '__main__':
    '''
    example use case
    
    '''
    RAW_DATA_DIR = './cnn_dailymail'
    train_fname = RAW_DATA_DIR + '/train.csv'
    test_fname = RAW_DATA_DIR + '/train.csv'
    valid_fname = RAW_DATA_DIR + '/train.csv'
    train_pd = pd.read_csv(train_fname)[:100]
    print('article : \n' + train_pd.iloc[0]['article'])
    print('summary : \n' + train_pd.iloc[0]['summary'])

    # pick a random sample and convert to bert cls_tokenid
    idx = random.randint(0, train_pd.shape[0] - 1)
    print('idx = ', idx)
    # idx=0
    raw_article0, raw_summary0 = train_pd.iloc[idx]['article'], train_pd.iloc[idx]['highlights']
    rdt = RawDocTokenize(raw_article0, raw_summary0,
                         article_max_token_length=512,
                         summary_max_token_length=512
                         )
    preprcessed1 = rdt.get_tokenized_output(padding=False)
    #preprcessed1 is a dictionary with keys: ['src', 'tgt', 'segs', 'clss']
    for k, v in preprcessed1.items():
        print(k, v)