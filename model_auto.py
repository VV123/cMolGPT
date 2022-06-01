import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim import SGD, Adam
from torch.nn import MSELoss, L1Loss
from torch.nn.init import xavier_uniform_
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from transformers import BertTokenizer, BertModel
import numpy as np
import sys
from utils import top_k_top_p_filtering, open_file, read_csv_file, load_sets
import vocabulary as mv
import dataset as md
import torch.utils.data as tud
import os.path
import glob
import math
import torch
import torch.nn as nn
from collections import Counter
from torch import Tensor
import io
import time
from torch.nn import (TransformerEncoder, TransformerDecoder,
                      TransformerEncoderLayer, TransformerDecoderLayer)

PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
########################
# x1 -> y
#######################
class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int,
                 emb_size: int, src_vocab_size: int, tgt_vocab_size: int,
                 dim_feedforward:int = 512, dropout:float = 0.1, args = None):
        super(Seq2SeqTransformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=emb_size, nhead=args.nhead,
                                                dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=args.nhead,
                                                dim_feedforward=dim_feedforward)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        

        tgt_vocab_size = src_vocab_size        
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        #self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.emb = nn.Embedding(4, dim_feedforward, padding_idx=0)

    def forward(self, trg: Tensor, tgt_mask: Tensor, tgt_padding_mask: Tensor, target: Tensor):
        #src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        #memory = self.transformer_encoder(src_emb, src_mask, src_padding_mask)
        s, b = trg.size()
        #memory = torch.zeros(s, b, 512).to('cuda')
        #print(target.size())
        memory = self.emb(target).unsqueeze(0).repeat(s, 1, 1)
        #print(memory.size())
        outs = self.transformer_decoder(tgt_emb, memory, tgt_mask, None,
                                        tgt_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer_encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, tgt_mask: Tensor, target: Tensor):
        s, b = tgt.size()
        memory = self.emb(target).unsqueeze(0).repeat(s, 1, 1)
        return self.transformer_decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)



######################################################################
# Text tokens are represented by using token embeddings. Positional
# encoding is added to the token embedding to introduce a notion of word
# order.
# 

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + 
                            self.pos_embedding[:token_embedding.size(0),:])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


######################################################################
# We create a ``subsequent word`` mask to stop a target word from
# attending to its subsequent words. We also create masks, for masking
# source and target padding tokens
# 

def generate_square_subsequent_mask(sz, DEVICE='cuda'):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(tgt, DEVICE='cuda'):
  tgt_seq_len = tgt.shape[0]
  tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
  tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
  return tgt_mask, tgt_padding_mask




