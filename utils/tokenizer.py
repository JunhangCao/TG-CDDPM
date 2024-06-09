import json
import os

import torch
import datasets
import numpy as np
from datasets import Dataset as Dataset2
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class TextTokenizer():
    def __init__(self, max_len=50):
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased',
                                                       cache_dir='../checkpoints/bert/scibert_scivocab_uncased')

    def encode(self, text):
        token = self.tokenizer.tokenize(text)
        input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(token))

        # padding
        if len(input_ids) < self.max_len:
            padding_zero = torch.zeros(self.max_len - len(input_ids))
            input_ids = torch.cat((input_ids, padding_zero))
        # truncation
        elif len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]

        return input_ids

    def batch_encode(self, texts):
        texts_input_ids = self.tokenizer(texts)['input_ids']
        padded_input_ids = []
        for i, input_ids in enumerate(texts_input_ids):
            if len(input_ids) < self.max_len:
                padding_zero = np.zeros(self.max_len - len(input_ids))
                padded_input_ids.append(np.concatenate((input_ids, padding_zero)))
            elif len(input_ids) >= self.max_len:
                padded_input_ids.append(np.array(input_ids[:self.max_len]))
        return torch.tensor(padded_input_ids, dtype=torch.int32)


class PepTokenizer():
    def __init__(self, vocab_path, max_len=50):
        vocab_dict = {}
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for row in f:
                t = row.strip().split('\t')[0]
                if t not in vocab_dict.keys():
                    vocab_dict[t] = len(vocab_dict) + 1
        self.tokenizer = vocab_dict
        self.rev_tokenizer = {v: k for k, v in vocab_dict.items()}
        # self.sep_token_id = vocab_dict['[SEP]']
        self.pad_token_id = 0
        self.vocab_size = len(self.tokenizer)
        self.max_len = max_len

    def batch_encode(self, sentences):
        input_ids = [[self.tokenizer[x] for x in seq] for seq in sentences]
        padded_input_ids = []
        for i, indices in enumerate(input_ids):
            if len(indices) < self.max_len:
                padding_zero = np.zeros(self.max_len - len(indices))
                padded_input_ids.append(np.concatenate((indices, padding_zero)))
            elif len(indices) >= self.max_len:
                padded_input_ids.append(np.array(indices[:self.max_len]))
        return torch.tensor(padded_input_ids, dtype=torch.int32)

def one_hot(seq, vocab, max_len=50):
    arr = np.zeros((max_len, 21))
    for i, w in enumerate(seq):
        arr[i][vocab[w]+1] = 1
    return list(arr)


def load_model_emb(args, vocab_size):
    ### random emb or pre-defined embedding like glove embedding. You can custome your own init here.
    model = torch.nn.Embedding(vocab_size, args.n_embd)
    path_save = '{}/random_emb.torch'.format(args.checkpoint_dir)
    path_save_ind = path_save + ".done"
    if os.path.exists(path_save):
        print('reload the random embeddings', model)
        model.load_state_dict(torch.load(path_save))
    else:
        print('initializing the random embeddings', model)
        torch.nn.init.normal_(model.weight)
        torch.save(model.state_dict(), path_save)
        with open(path_save_ind, "x") as _:
            pass
    print('reload the random embeddings', model)
    model.load_state_dict(torch.load(path_save))

    return model

class TextDataset(Dataset):

    def __init__(self, text_datasets, model_emb=None):
        super().__init__()
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets['train'])
        self.model_emb = model_emb

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with torch.no_grad():
            input_ids = self.text_datasets['train'][idx]['input_ids']
            hidden_state = self.model_emb(torch.tensor(input_ids).long())

            # obtain the input vectors, only used when word embedding is fixed (not trained end-to-end)
            arr = np.array(hidden_state, dtype=np.float32)

            out_kwargs = {}
            out_kwargs['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
            out_kwargs['self_condition'] = np.array(self.text_datasets['train'][idx]['label'])
            return arr, out_kwargs


class tokenizer():
    def __init__(self, vocab_path):
        vocab_dict = {}
        with open(vocab_path, 'r', encoding='utf-8') as f:
            for row in f:
                t = row.strip().split('\t')[0]
                if t not in vocab_dict.keys():
                    vocab_dict[t] = len(vocab_dict)
        self.tokenizer = vocab_dict
        self.rev_tokenizer = {v: k for k, v in vocab_dict.items()}
        # self.sep_token_id = vocab_dict['[SEP]']
        self.pad_token_id = 0
        self.vocab_size = len(self.tokenizer)

    def encode_token(self, sentences):
        input_ids = [[self.tokenizer.get(x)+1 for x in seq] for seq in sentences]
        # input_ids = [one_hot(seq, self.tokenizer) for seq in sentences]
        return input_ids

    def decode_token(self, seq):
        if len(seq.shape) > 1:
            seq = seq.squeeze(-1).tolist()
        tokens = ""
        for id in seq:
            if id == self.pad_token_id:
                tokens += ""
                continue
            tokens += self.rev_tokenizer[id - 1]
        # while len(seq) > 0 and seq[-1] == self.pad_token_id:
        #     seq.pop()
        # tokens = " ".join([self.rev_tokenizer[x] for x in seq]).replace('__ ', '').replace('@@ ', '')
        return tokens

    def decode_batch_token(self, batch):
        batch_token = []
        for b in batch:
            seq = b.squeeze(-1).tolist()
            tokens = ""
            for id in seq:
                if id == self.pad_token_id:
                    tokens += ""
                    continue
                tokens += self.rev_tokenizer[id - 1]
            # while len(seq) > 0 and seq[-1] == self.pad_token_id:
            #     seq.pop()
            # tokens = " ".join([self.rev_tokenizer[x] for x in seq]).replace('__ ', '').replace('@@ ', '')
            batch_token.append([tokens.replace(' ','')])
        return batch_token


def load_data(pep_tokenizer, seq_len, tag, args):
    if tag == 'train':
        path = '../dataset/backup/ensemble_train.jsonl'
    elif tag == 'test':
        path = '../dataset/backup/ensemble_test.jsonl'
    else:
        raise ValueError()
    sentence = {'src': [], 'trg': []}
    with open(path, 'r', encoding='utf-8') as reader:
        for row in reader:
            sentence['src'].append(json.loads(row)['src'])
            seq = json.loads(row)['trg']
            s = ""
            for i in range(len(seq) - 1):
                s += seq[i] + ' '
            s += seq[-1]
            sentence['trg'].append(s)

    raw_datasets = Dataset2.from_dict(sentence)

    def tokenize_function(examples):
        # input_id_y = vocab_dict.encode_token(examples['trg'])
        input_id_y = pep_tokenizer(examples['trg'],
                                padding=True,
                                truncation=True,
                                return_tensors="pt",
                                max_length=50)['input_ids']
        text_tokenizer = TextTokenizer()
        target = text_tokenizer.batch_encode(examples['src'])
        result_dict = {'input_ids': input_id_y, 'label': target}

        return result_dict

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=1,
        remove_columns=['src', 'trg'],
        load_from_cache_file=True,
        desc="Running tokenizer on public_database",
    )

    def pad_function(group_lst):
        max_length = seq_len
        group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], pep_tokenizer.pad_token_id, max_length)
        # group_lst['input_mask'] = _collate_batch_helper(group_lst['input_mask'], 1, max_length)
        return group_lst

    lm_datasets = tokenized_datasets.map(
        pad_function,
        batched=True,
        num_proc=1,
        desc=f"padding",
    )

    raw_datasets = datasets.DatasetDict()
    raw_datasets['train'] = lm_datasets

    model_emb = load_model_emb(args, pep_tokenizer.vocab_size)
    # model_emb = nn.Embedding(vocab_dict.vocab_size+1, seq_len)
    dataset = TextDataset(
        raw_datasets,
        model_emb=model_emb
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=True
    )
    while True:
        yield from dataloader


def one_hot_encode(array, cls):
    ret = torch.zeros((len(array), cls))
    for i in range(len(array)):
        ret[i][array[i]] = 1
    return ret.long()

def infinite_loader(data_loader):
    while True:
        yield from data_loader


def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):
    result = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    mask_ = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result