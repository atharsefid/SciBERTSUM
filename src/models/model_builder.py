import torch
import torch.nn as nn
from models.modeling_bert import BertModel, BertConfig
from torch.nn.init import xavier_uniform_
from typing import Optional, Tuple
from models.encoder import Classifier, ExtTransformerEncoder
from models.longExtractiveFormer import LongExtTransformerEncoder, LongFormerConfig
from models.optimizers import Optimizer
from others.log import logger
from torch.nn import functional as F
from torch import Tensor, device


def build_optim(args, model, checkpoint):
    """ Build optimizer """

    if checkpoint is not None:
        optim = checkpoint['optim'] #### fix it already had '[0]' at the end of this line
        saved_optimizer_state_dict = optim.optimizer.state_dict()
        optim.optimizer.load_state_dict(saved_optimizer_state_dict)
        if args.visible_gpus != '-1':
            for state in optim.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()

        if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
            raise RuntimeError(
                "Error: loaded Adam optimizer from existing model" +
                " but optimizer state is empty")

    else:
        optim = Optimizer(
            args.optim, args.lr, args.max_grad_norm,
            beta1=args.beta1, beta2=args.beta2,
            decay_method='noam',
            warmup_steps=args.warmup_steps)

    optim.set_parameters(list(model.named_parameters()))

    return optim


class Bert(nn.Module):
    def __init__(self, large, temp_dir, finetune=False):
        super(Bert, self).__init__()
        if large:
            self.model = BertModel.from_pretrained('bert-large-uncased', cache_dir=temp_dir)
        else:
            self.model = BertModel.from_pretrained('bert-base-uncased', cache_dir=temp_dir)

        self.finetune = finetune

    def forward(self, x, token_sections, segs, mask):
        if self.finetune:
            top_vec, _ = self.model(x, token_sections, segs, attention_mask=mask)
        else:
            self.eval()
            with torch.no_grad():
                top_vec, _ = self.model(x, token_sections, segs, attention_mask=mask)
        return top_vec


class ExtSummarizer(nn.Module):
    def __init__(self, args, device, checkpoint):
        super(ExtSummarizer, self).__init__()
        self.args = args
        self.device = device
        self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)
        self.doc_len = args.max_pos
        self.config = LongFormerConfig(hidden_size=self.bert.model.config.hidden_size,
                                       intermediate_size=args.ext_ff_size,
                                       num_hidden_layers=args.ext_layers,
                                       num_attention_heads=args.ext_heads,
                                       hidden_dropout_prob=args.ext_dropout)
        self.ext_layer = LongExtTransformerEncoder(self.config)

        if self.doc_len > 512:
            my_pos_embeddings = nn.Embedding(self.doc_len, self.bert.model.config.hidden_size)
            my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
            my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None,
                                                  :].repeat(self.doc_len - 512, 1)
            self.bert.model.embeddings.position_embeddings = my_pos_embeddings

        if checkpoint:
            self.load_state_dict(checkpoint['model'], strict=True)
        else:
            if args.param_init != 0.0:
                for p in self.ext_layer.parameters():
                    p.data.uniform_(-args.param_init, args.param_init)
            if args.param_init_glorot:
                for p in self.ext_layer.parameters():
                    if p.dim() > 1:
                        xavier_uniform_(p)
        print('device is::::   ', device)
        self.to(device)

    def forward(self, src, sections, token_sections, segs, clss, mask_src, mask_cls):
        print('-'*200)
        # batch size must be 1
        print('src:', src.shape)
        print('segs:', segs.shape)
        print('clss:', clss.shape)
        print('mask_src:', mask_src.shape)
        print('mask_cls:', mask_cls.shape)
        print('section shape:', sections.shape)
        print(' token_sections:', token_sections.shape)

        top_vec = self.bert(src, token_sections, segs, mask_src)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
        sents_vec = sents_vec * mask_cls[:, :, None].float()
        print('-------')
        print('sents_vec:', sents_vec.shape)
        print('sent_vec masks:', mask_cls.shape)
        # ###################################################################################
        # prepare sents_vec for long former
        inputs_embeds = sents_vec
        attention_mask = mask_cls
        input_shape = sents_vec.size()[:-1]
        # todo generate global attention indices
        global_attention_mask = None  # fix
        # merge `global_attention_mask` and `attention_mask`
        if global_attention_mask is not None:
            attention_mask = self._merge_to_attention_mask(attention_mask, global_attention_mask)
        token_type_ids = segs
        position_ids = None
        print('inputs_embeds before padding:', inputs_embeds.shape)
        padding_len, inputs_embeds, attention_mask, token_type_ids, position_ids = self._pad_to_window_size(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            pad_token_id=self.config.pad_token_id,
        )
        print('inputs_embeds after padding:', inputs_embeds.shape)
        assert (
                inputs_embeds.shape[1] % self.config.attention_window[0] == 0
        ), f"padded inputs_embeds of size {inputs_embeds.shape[1]} is not a multiple of window size {self.config.attention_window}"

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)[:, 0, 0, :]

        sent_scores = self.ext_layer(inputs_embeds, attention_mask, extended_attention_mask).squeeze(-1)
        return sent_scores, extended_attention_mask

    def _merge_to_attention_mask(self, attention_mask: torch.Tensor, global_attention_mask: torch.Tensor):
        # longformer self attention expects attention mask to have 0 (no attn), 1 (local attn), 2 (global attn)
        # (global_attention_mask + 1) => 1 for local attention, 2 for global attention
        # => final attention_mask => 0 for no attention, 1 for local attention 2 for global attention
        if attention_mask is not None:
            attention_mask = attention_mask * (global_attention_mask + 1)
        else:
            # simply use `global_attention_mask` as `attention_mask`
            # if no `attention_mask` is given
            attention_mask = global_attention_mask + 1
        return attention_mask

    def _pad_to_window_size(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,  pad_token_id: int):
        """A helper function to pad tokens and mask to work with implementation of Longformer self-attention."""
        # padding
        attention_window = (
            self.config.attention_window
            if isinstance(self.config.attention_window, int)
            else max(self.config.attention_window)
        )

        assert attention_window % 2 == 0, f"`attention_window` should be an even value. Given {attention_window}"
        input_shape = inputs_embeds.shape
        batch_size, seq_len = input_shape[:2]

        padding_len = (attention_window - seq_len % attention_window) % attention_window
        if padding_len > 0:
            logger.info(
                "Input ids are automatically padded from {} to {} to be a multiple of `config.attention_window`: {}".format(
                    seq_len, seq_len + padding_len, attention_window
                )
            )
            if position_ids is not None:
                # pad with position_id = pad_token_id as in modeling_roberta.RobertaEmbeddings
                position_ids = F.pad(position_ids, (0, padding_len), value=pad_token_id)
            if inputs_embeds is not None:
                input_ids_padding = inputs_embeds.new_full(
                    (batch_size, padding_len),
                    self.config.pad_token_id,
                    dtype=torch.long,
                )
                inputs_embeds_padding = self.bert.model.embeddings(input_ids_padding)
                inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_padding], dim=-2)

            attention_mask = F.pad(attention_mask, (0, padding_len), value=False)  # no attention on the padding tokens
            token_type_ids = F.pad(token_type_ids, (0, padding_len), value=0)  # pad with token_type_id = 0

        return padding_len, inputs_embeds, attention_mask, token_type_ids, position_ids

    def get_extended_attention_mask(self, attention_mask: Tensor, input_shape: Tuple[int], device: device) -> Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
        converts 0 1 2 mask labels to -10000 , 0 , 10000 .
        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                # casual mask for a sequence of length 4:
                # T F F F
                # T T F F
                # T T T F
                # T T T T

                # in case past_key_values are used we need to add a prefix ones mask to the causal mask
                # causal and attention masks must have same type with pytorch version < 1.3
                causal_mask = causal_mask.to(attention_mask.dtype)

                if causal_mask.shape[1] < attention_mask.shape[1]:
                    prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
                    causal_mask = torch.cat(
                        [
                            torch.ones(
                                (batch_size, seq_length, prefix_seq_len), device=device, dtype=causal_mask.dtype
                            ),
                            causal_mask,
                        ],
                        axis=-1,
                    )

                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask