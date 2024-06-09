import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import warnings
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer, AutoModel, AutoConfig
from config.backen_config import ProGenConfig
from model.backend import ProGenForCausalLM
from config.base_config import add_dict_to_argparser
from info_nce import InfoNCE
from einops import reduce, rearrange

from utils.tokenizer import TextTokenizer

warnings.filterwarnings('ignore')


def create_argparser():
    defaults = dict(
        batch_size=32,
        clip_epoches=50,
        fac_epoches=10,
        lr=1e-4,
        weight_decay=0.999,
        valid_rate=0.2,
        vocab_size=21,
        n_positions=256,
        n_ctx=256,
        n_embd=256,
        n_layer=2,
        seq_len=[50],
        class_cond=True,
        vocab_path="../mapping/vocab.txt",
        save_path="../checkpoints/"
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def get_text_peptides(path):
    text_peptides = pd.read_csv(path)
    des = list(text_peptides['Description'])
    sequences = list(text_peptides['Sequence'])
    texts = []
    seqs = []
    for i in range(len(des)):
        text = str(des[i])
        texts.append(text)
        s = ""
        for j in range(len(sequences[i]) - 1):
            s += sequences[i][j] + ' '
        s += sequences[i][-1]
        seqs.append(s)
    return texts, seqs


def l2norm(t):
    return F.normalize(t, dim=-1)


def matrix_diag(t):
    device = t.device
    i, j = t.shape[-2:]
    num_diag_el = min(i, j)
    i_range = torch.arange(i, device=device)
    j_range = torch.arange(j, device=device)
    diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j')
    diag_el = t.masked_select(diag_mask)
    return rearrange(diag_el, '(b d) -> b d', d=num_diag_el)

def masked_mean(t, mask, dim=1, eps=1e-6):
    t = t.masked_fill(~mask, 0.)
    numer = t.sum(dim=dim)
    denom = mask.sum(dim=dim).clamp(min=eps)
    return numer / denom

def max_neg_value(dtype):
    return -torch.finfo(dtype).max

def log(t, eps = 1e-20):
    return torch.log(t + eps)


class TextEncoder(nn.Module):
    def __init__(self,
                 hidden_state_dim=256,
                 output_dim=256,
                 max_len=50):
        super().__init__()
        self.bert_config = AutoConfig.from_pretrained('allenai/scibert_scivocab_uncased',
                                                      cache_dir='../checkpoints/bert/scibert_scivocab_uncased')
        self.sci_bert = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased',
                                                  cache_dir='../checkpoints/bert/scibert_scivocab_uncased')
        # self.ln = nn.Sequential(
        #     nn.LayerNorm(self.bert_config.hidden_size),
        #     nn.Linear(self.bert_config.hidden_size, hidden_state_dim),)
        self.ln = nn.Linear(self.bert_config.hidden_size, output_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_state_dim * max_len, hidden_state_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_state_dim),
            nn.Linear(hidden_state_dim, output_dim),
        )

    def get_features(self, x):
        return self.ln(self.sci_bert(x)['last_hidden_state'])

    def forward(self, x):
        b = x.shape[0]
        hidden_states = self.sci_bert(x)
        hidden_states = hidden_states['last_hidden_state']
        hidden_states = self.ln(hidden_states)
        hidden_states = hidden_states.reshape(b, -1)
        hidden_states = self.mlp(hidden_states)
        return hidden_states


class PepEncoder(nn.Module):
    def __init__(self, config, max_len=50):
        super().__init__()
        # self.pep_encoder = AutoModel.from_pretrained("Rostlab/prot_bert",
        #                                              cache_dir='../checkpoints/bert/prot_bert')
        self.pep_encoder = ProGenForCausalLM(config)
        self.ln = nn.Sequential(
            nn.Linear(config.n_embd*max_len, config.n_embd),
            nn.GELU(),
            nn.LayerNorm(config.n_embd),
            nn.Linear(config.n_embd, config.n_embd),
        )

    def get_features(self, x, input_embeds=None):
        if input_embeds is not None:
            return self.pep_encoder(inputs_embeds=input_embeds)
        else:
            return self.pep_encoder(x)

    def forward(self, x, input_embeds=None):
        if input_embeds is not None:
            b = input_embeds.shape[0]
            hidden_states = self.pep_encoder(inputs_embeds=input_embeds)
        else:
            b = x.shape[0]
            hidden_states = self.pep_encoder(x)
        hidden_states = hidden_states.reshape(b, -1)
        hidden_states = self.ln(hidden_states)
        return hidden_states


class Facilitator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mlp = ProGenForCausalLM(config)

    def forward(self, x):
        output = self.mlp(inputs_embeds=x)
        return output


class BatchDataset(Dataset):
    def __init__(self, sentences):
        super().__init__()
        self.raw_dataset = sentences
        self.len = len(sentences['src'])

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        text = self.raw_dataset['src'][idx]
        # print(self.raw_dataset['train'][idx]['src'])
        peptide = self.raw_dataset['trg'][idx]
        return text, peptide


if __name__ == '__main__':
    # data loading
    train_texts, train_peptides = get_text_peptides('../dataset/backup/ensemble_train.csv')
    test_texts, test_peptides = get_text_peptides('../dataset/backup/ensemble_test.csv')
    args = create_argparser().parse_args()
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    text_tokenizer = TextTokenizer()
    text_encoder = TextEncoder().to(device)
    # text_encoder.load_state_dict(torch.load(args.save_path + 'text_encoder.pt'))
    pep_tokenizer = AutoTokenizer.from_pretrained('../checkpoints/bert/prot_bert')
    pep_config = ProGenConfig(vocab_size=pep_tokenizer.vocab_size,
                              n_positions=args.n_positions,
                              n_ctx=args.n_ctx,
                              n_embd=args.n_embd,
                              n_layer=args.n_layer)
    # pep_tokenizer = PepTokenizer(args.vocab_path)

    pep_encoder = PepEncoder(pep_config).to(device)
    # pep_encoder.load_state_dict(torch.load(args.save_path + 'pep_encoder.pt'))
    # train
    text_tokens = text_tokenizer.batch_encode(train_texts)
    pep_tokens = pep_tokenizer(train_peptides,
                               padding=True,
                               truncation=True,
                               return_tensors="pt",
                               max_length=50)['input_ids']
    sentence = {'src': [], 'trg': []}
    for i in range(len(text_tokens)):
        sentence['src'].append(text_tokens[i])
        sentence['trg'].append(pep_tokens[i])
    dataset = BatchDataset(sentence)

    train_dataloader = DataLoader(dataset,
                                  shuffle=True,
                                  num_workers=0,
                                  batch_size=args.batch_size,
                                  drop_last=True, )
    # valid
    test_text_tokens = text_tokenizer.batch_encode(test_texts)
    test_pep_tokens = pep_tokenizer(test_peptides,
                               padding=True,
                               truncation=True,
                               return_tensors="pt",
                               max_length=50)['input_ids']
    valid = {'src': [], 'trg': []}
    for i in range(len(test_pep_tokens)):
        valid['src'].append(test_text_tokens[i])
        valid['trg'].append(test_pep_tokens[i])
    valid_dataset = BatchDataset(valid)
    valid_dataloader = DataLoader(valid_dataset,
                                  shuffle=True,
                                  num_workers=0,
                                  batch_size=args.batch_size,
                                  drop_last=True, )
    infonce = InfoNCE()
    optimizer1 = torch.optim.Adam(params=text_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer2 = torch.optim.Adam(params=pep_encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # text_encoder.load_state_dict(torch.load('../checkpoints/text_encoder.pt'))
    # pep_encoder.load_state_dict(torch.load('../checkpoints/pep_encoder.pt'))

    print('-------------------training-----------------------')
    for e in range(args.clip_epoches):
        e_loss = 0
        for text, pep in train_dataloader:
            # optimizer3.zero_grad()
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            text = text.to(device)
            pep = pep.to(device)
            text_hidden_states = text_encoder(text)
            # pep_input_embeds = pep_encoder.pep_encoder.get_input_embeddings()(pep)
            pep_hidden_states = pep_encoder(pep)
            loss = infonce(text_hidden_states, pep_hidden_states)
            e_loss += loss
            loss.backward()
            # optimizer3.step()
            optimizer1.step()
            optimizer2.step()
            torch.cuda.empty_cache()
        print("epoch :{}, clip training loss: {}".format(e + 1, e_loss / len(train_dataloader)))

        valid_loss = 0
        for valid_text, valid_pep in valid_dataloader:
            with torch.no_grad():
                valid_text = valid_text.to(device)
                valid_pep = valid_pep.to(device)
                valid_text_hidden_states = text_encoder(valid_text)
                valid_pep_hidden_states = pep_encoder(valid_pep)
                valid_text_hidden_states = valid_text_hidden_states.reshape(args.batch_size, -1)
                valid_pep_hidden_states = valid_pep_hidden_states.reshape(args.batch_size, -1)
                # valid_text_z = facilitator(valid_text_z)
                # valid_pep_z = facilitator(valid_pep_z)

                loss = infonce(valid_text_hidden_states, valid_pep_hidden_states)
                # loss2 = loss_fn(valid_pep_z, valid_text_z)
                # loss = (loss1 + loss2) / 2
                valid_loss += loss
        print("epoch :{}, clip valid loss: {}".format(e + 1, valid_loss / len(valid_dataloader)))
    # save model parameters
    torch.save(text_encoder.state_dict(), args.save_path + 'text_encoder.pt')
    torch.save(pep_encoder.state_dict(), args.save_path + 'pep_encoder.pt')

    print('--------------------done--------------------------')
