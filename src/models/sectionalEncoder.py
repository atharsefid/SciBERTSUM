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
        sec_start_index = 0
        section_attentions = []
        ########################## sectional attentions ###############
        for sec_vecs in sections_sent_vecs:
            n_sents, hidden_size = sec_vecs.shape
            pos_emb = self.pos_emb.pe[:, :n_sents]
            x = sec_vecs + pos_emb
            # print('sections_sent_vecs ::::, pos embed shape', sec_vecs.shape, pos_emb.shape, x.shape, mask[:,sec_start_index:sec_start_index+n_sents].shape)
            for i in range(self.config.num_hidden_layers):
                x = self.transformer_inter[i](i, x, x, ~ mask[:,sec_start_index:sec_start_index+n_sents])  # all_sents * max_tokens * dim
            sec_start_index += n_sents
            section_attentions.append(x)
        ############################# full attentions #################
        all_sent_vecs = torch.cat(sections_sent_vecs, 0)
        n_sents, hidden_size = all_sent_vecs.shape
        pos_emb = self.pos_emb.pe[:, :n_sents]
        doc_attentions = all_sent_vecs + pos_emb
        for i in range(self.config.num_hidden_layers):
            doc_attentions = self.transformer_inter[i]\
                (i, doc_attentions, doc_attentions, ~ mask)  # all_sents * max_tokens * dim
        ##############################################################

        sectional_attentions = torch.cat(section_attentions, dim=1)
        attentions = sectional_attentions +  doc_attentions
        # print('final attentions', attentions.shape)
        attentions = self.layer_norm(attentions)
        sent_scores = self.sigmoid(self.wo(attentions))
        sent_scores = sent_scores.squeeze(-1) * mask.float()

        return sent_scores
