import torch
import torch.nn as nn
import torch as t
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np




class SentenceExtractor(nn.Module):
    def __init__(self, config ):
        super(SentenceExtractor, self).__init__()

        self.input_size = config.hidden_size
        self.hidden_size = config.hidden_size
        self.layers = 1
        self.bidirectional = False

        self.rnn = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.layers,
                          batch_first=True,
                          bidirectional=self.bidirectional)

        self.output_size = 100
        self.semantic_layer = nn.Linear(self.hidden_size, self.output_size)
        self.position_layer = nn.Linear(self.hidden_size, self.output_size)
        self.section_layer = nn.Linear(self.hidden_size, self.output_size)
        self.context_layer = nn.Linear(self.hidden_size, self.output_size)
        self.length_layer = nn.Linear(self.hidden_size, self.output_size)

        self.final_layer = nn.Linear(6 * self.output_size, 1)


        self.semantic_dropout = nn.Dropout(0.5)
        self.position_dropout = nn.Dropout(0.2)
        self.sections_dropout = nn.Dropout(0.2)
        self.context_dropout = nn.Dropout(0.2)
        self.length_dropout = nn.Dropout(0.2)

        self.sent_correlation_weight = torch.nn.Parameter(torch.zeros(self.hidden_size, self.hidden_size))
        torch.nn.init.xavier_uniform(self.sent_correlation_weight)
        self.correlation_layer = nn.Linear(self.hidden_size, self.output_size)
        self.correlation_dropout = nn.Dropout(0.2)



    def forward(self, sentence_embeddings, position_embedding, section_embedding,context_embedding, length_embedding, doc_embedding):
        """
        :param sentence_embeddings: batch_size, num_sents, hidden_size
        :param position_embeddings: batch_size, num_sents, hidden_size
        :param section_embeddings: batch_size, num_sents, hidden_size
        :param doc_embedding: batch_size, hidden_size
        :return: logits. shape = batch_size, num_sents
        """

        [_, seq_len, _] = sentence_embeddings.size()

        # even after batch_first=True, the h0 follows [num_layers * num_directions, batch, hidden_size]
        doc_embedding = doc_embedding.unsqueeze(1).transpose(0, 1).contiguous()
        h_t, _ = self.rnn(sentence_embeddings, doc_embedding)  # batch, seq_len, self.hidden_size

        semantic_embedding = h_t.contiguous().view(-1, self.hidden_size)
        semantic_embed = torch.relu(self.semantic_dropout(self.semantic_layer(semantic_embedding)))
        position_embed = torch.relu(self.position_dropout(self.position_layer(position_embedding)))
        section_embed = torch.relu(self.sections_dropout(self.section_layer(section_embedding)))
        context_embed = torch.relu(self.context_dropout(self.context_layer(context_embedding)))
        length_embed = torch.relu(self.length_dropout(self.length_layer(length_embedding)))

        # correlation between sentences

        sent_embed_s = sentence_embeddings.squeeze(0)
        correlation_weight  = torch.tanh(torch.matmul(torch.matmul(sent_embed_s, self.sent_correlation_weight) ,torch.transpose(sent_embed_s, 1, 0)))
        # this line is to identify sentences that are correlated to other sentences (they have shared words with other sents). So better to exclude them.
        correlation_embed = torch.relu(self.correlation_dropout(self.correlation_layer(torch.matmul(correlation_weight, sent_embed_s))))
        sentence_features = torch.cat( [semantic_embed, position_embed, section_embed.squeeze(0), length_embed.squeeze(0), context_embed.squeeze(0), correlation_embed], dim =1)
        scores = self.final_layer(torch.relu(sentence_features))
        return scores



class DocumentEncoder(nn.Module):

    def __init__(self, batch_size, sent_hidden, word_gru_hidden):

        super(DocumentEncoder, self).__init__()
        self.batch_size = batch_size
        self.sent_hidden = sent_hidden
        self.word_gru_hidden = word_gru_hidden
        self.sent_encoder = nn.Linear(sent_hidden, sent_hidden)
        self.attn_encoder = nn.Linear(sent_hidden, 1)
        self.sent_state = nn.Parameter(torch.zeros(2, self.batch_size, self.sent_hidden))
        self.softmax_sent = nn.Softmax(dim=1)


    def forward(self, sentence_vectors):
        sent_squish = self.sent_encoder(sentence_vectors)
        sent_attn = self.attn_encoder(sent_squish)
        sent_attn_norm = self.softmax_sent(sent_attn)
        sent_attn_vectors = self.attention_mul(sentence_vectors, sent_attn_norm)
        return  sent_attn_vectors

    @staticmethod
    def attention_mul(sentence_vectors, att_weights):
        att_expand = att_weights.expand_as(sentence_vectors)
        attn_vectors = torch.einsum("bsd,bsd->bsd", (sentence_vectors, att_expand))
        return_value = torch.mean(attn_vectors, 1)
        return return_value


test = False
if test:
    sentence_vectors = torch.Tensor(2,3,10)
    DE = DocumentEncoder(2,10, 10)
    doc_vector =DE(sentence_vectors)
    print('document vector:::', doc_vector.shape)

    class Config:
        def __init__(self):
            self.hidden_size= 10

    c = Config()

    extractor = SentenceExtractor(c)
    logits  = extractor(sentence_vectors,doc_vector)
    print('logits::::', logits.shape)