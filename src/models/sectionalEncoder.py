import math

import torch
import torch.nn as nn

from models.neural import MultiHeadedAttention
from models.longExtractiveFormer import PositionalEncoding, PositionwiseFeedForward

class Classifier(nn.Module):
    def __init__(self, hidden_size):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask_cls):
        h = self.linear1(x).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_cls.float()
        return sent_scores


class SectionalTransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super(SectionalTransformerEncoderLayer, self).__init__()
        self.config = config
        self.self_attn = MultiHeadedAttention(self.config)
        self.feed_forward = PositionwiseFeedForward(self.config)
        self.layer_norm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)


    def forward(self, iter, query, inputs, mask):
        if iter != 0:
            input_norm = self.layer_norm(inputs)
        else:
            input_norm = inputs

        mask = mask.unsqueeze(1)
        context = self.self_attn(input_norm, input_norm, input_norm, mask=mask)
        out = self.dropout(context) + inputs
        return self.feed_forward(out)


class SectionalExtTransformerEncoder(nn.Module):

    def __init__(self, config):
        super(SectionalExtTransformerEncoder, self).__init__()
        self.config = config
        self.pos_emb = PositionalEncoding(self.config)
        self.transformer_inter = nn.ModuleList(
            [SectionalTransformerEncoderLayer(self.config) for _ in range(self.config.num_hidden_layers)])
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.layer_norm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
        self.wo = nn.Linear(self.config.hidden_size, 1, bias=True)
        self.sigmoid = nn.Sigmoid()


    def forward(self, sections_sent_vecs, sections, mask):
        """ See :obj:`EncoderBase.forward()`"""
        # get each section and apply attention inside each sections


        batch_size, n_sents = top_vecs.size(0), top_vecs.size(1)
        pos_emb = self.pos_emb.pe[:, :n_sents]
        x = top_vecs * mask[:, :, None].float()
        x = x + pos_emb

        for i in range(self.config.num_hidden_layers):
            x = self.transformer_inter[i](i, x, x, ~ mask)  # all_sents * max_tokens * dim

        x = self.layer_norm(x)
        sent_scores = self.sigmoid(self.wo(x))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores
