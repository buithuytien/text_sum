import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import pandas as pd
import numpy as np
import h5py
import pickle
import os
import sys
from torch.utils.data.dataloader import default_collate

# test automatically upload

np.set_printoptions(threshold=sys.maxsize)
import torch; print(torch.__version__)

class SummarisationDataset(Dataset):
    def __init__(self, path, padding_token = -1, subset=None):
        if path.endswith('.hdf5'):
            self.dataset = h5py.File(path, 'r')
            self.keys = list(self.dataset.keys())  # ['src', 'tgt', 'src_sent_labels', 'segs', 'clss']
            self.padding_token = padding_token
    def __len__(self):
        return self.dataset[self.keys[-1]].shape[0]

    def __getitem__(self, idx):
        out = {}
        for k in self.keys:
            out[k] = self.dataset[k][idx, :]
            #         out['token_len'] = np.count_nonzero(self.dataset[k][idx, :]+1)
            out['token_len_src'] = np.sum(self.dataset['src'][idx, :] != self.padding_token)
            out['token_len_tgt'] = np.sum(self.dataset['tgt'][idx, :] != self.padding_token)
            out['sent_len'] = np.sum(self.dataset['clss'][idx, :] != self.padding_token)
            out['padding_token'] = self.padding_token
        return out


def collate_fn(batch_list):
    '''
    batch_list = list of dictionary retrieved from SummarisationDataset
    data[i].keys() = dict_keys(['clss', 'segs', 'src', 'src_sent_labels', 'tgt'])

    return : dict()
    '''
    eos_token = 2 # end of stream token, unused1
    batch = default_collate(batch_list)
    max_src_token_len = max(batch['token_len_src'])
    max_tgt_token_len = max(batch['token_len_tgt'])
    max_sent_len = max(batch['sent_len'])
    # padding_token = batch_list['padding_token'][0]

    src_token_keys = ['src', 'segs']
    for k in src_token_keys:
        batch[k] = batch[k][:, :max_src_token_len].type(torch.LongTensor)

    tgt_token_keys = ['tgt']
    for k in tgt_token_keys:
        padded_tgt = batch[k][:, :max_tgt_token_len].type(torch.LongTensor)
        # insert unused token
        for i in range(padded_tgt.shape[0]):
            if batch['token_len_tgt'][i] == max_tgt_token_len:
                padded_tgt[i, -1] = eos_token
        batch[k] = padded_tgt

    sent_keys = ['clss', 'src_sent_labels']
    for k in sent_keys:
        batch[k] = batch[k][:, :max_sent_len].type(torch.LongTensor)

    # create labels
    # batch['label'] = batch['tgt'].clone()
    # batch['label'] = batch['label'].type(torch.LongTensor)
    # print(batch['src'].shape)
    return batch


def batch2BertPadId(token_id_tensor: torch.tensor, old_pad_token_id=-1, new_pad_token_id=0) -> torch.tensor:
    '''
    convert any value of old_pad_token_id into new_pad_token_id
    input: tensor of bert token id
    shape: batch_size * max_token_len. eg: 32*512
    '''
    token_id_tensor[token_id_tensor == old_pad_token_id] = new_pad_token_id
    return token_id_tensor


def createAttMask(token_id_tensor: torch.tensor, pad_token_id=0) -> torch.tensor:
    '''
    make attention mask for tensor of bert token id
    making every value == pad_token_id to 0, and
    making every value != pad_token_id to 1
    '''
    mask = token_id_tensor.clone()
    mask[mask == pad_token_id] = 0
    mask[mask != pad_token_id] = 1
    return mask


train_file = './preprocessed/train_data.hdf5'
train_dataset = SummarisationDataset(path = train_file)
batch_list = [train_dataset[idx] for idx in range(3)]
batch = collate_fn(batch_list)

for k, v in batch.items():
    print(k, v)
