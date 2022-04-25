import torch
from torch import nn, Tensor
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader, SequentialSampler
from torch.utils.data.dataloader import default_collate

from transformers import BertTokenizer, BertModel, BertConfig

import math
import numpy as np
import pandas as pd
import h5py
import pickle

import math
from typing import Tuple

bert_configuration = BertConfig()

class Bert(nn.Module):
    def __init__(self, finetune = False):
        super().__init__()
#         bert_configuration = BertConfig()
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.finetune = finetune
        self.config = self.model.config

    def forward(self, x, segs, mask):
        if(self.finetune):
            top_vec = self.model(input_ids = x,
                                token_type_ids  = segs,
                                attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                # thuy tien bui  test test2
                top_vec= self.model(input_ids = x,
                                    token_type_ids  = segs,
                                    attention_mask=mask)
        return top_vec


class EncoderDecoder(nn.Module):
    def __init__(self, num_tokens, dim_model, num_decoder_layers, dropout_p):
        super().__init__()
        self.bert = Bert()  # encoder
        bert_config = self.bert.config
        decoder_layer = nn.TransformerDecoderLayer(d_model=bert.config.hidden_size,
                                                   dropout=dropout_p)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.out = nn.Linear(dim_model, num_tokens)  # vocab size of bert

    def forward(self, src, tgt, src_mask=None, src_segs=None, tgt_mask=None,
                src_pad_mask=None, tgt_pad_mask=None):
        top_vec = self.bert(src, src_segs, src_mask)  # batch_size, n_token_in_sample, 768 . eg batch_size*512*768
        # TODO: add decoder layer


bert = Bert()

data_train1 = torch.load('../preprocessed/batch1.pt')

# for k,v in data_train1.items():
#     print(k, v)

max_token_len = 512
src = data_train1['src'][:,:max_token_len]
segs = data_train1['segs'][:,:max_token_len]
mask_src = data_train1['att_mask_src'][:,:max_token_len]

print(src.shape, segs.shape, mask_src.shape)
# assert src.shape[0] == seqs.shape[0] == mask_src.shape[0] == batch_size
print(mask_src)

encoder_output = bert(src,  segs, mask_src)
print('encoder_output', encoder_output)

top_vec = bert(src, segs, mask_src)
print(top_vec)


