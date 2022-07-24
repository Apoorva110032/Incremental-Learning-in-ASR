import os

import torch
import torch.nn as nn
import os

from omegaconf import DictConfig, OmegaConf, open_dict
from nemo.collections.asr.parts.submodules.conformer_modules import ConformerLayer
from nemo.collections.asr.data import audio_to_text_dataset
from nemo.collections.asr.parts.utils.activations import Swish
from nemo.collections.asr.parts.submodules.multi_head_attention import (
    MultiHeadAttention,
    RelPositionMultiHeadAttention,
)

__all__ = ['OverwrittenConformerLayer']


class OverwrittenConformerLayer(ConformerLayer):
    def __init__(self, d_model, d_ff, self_attention_model='rel_pos', n_heads=4, conv_kernel_size=31,
                 conv_norm_type='batch_norm', dropout=0.1, dropout_att=0.1, pos_bias_u=None, pos_bias_v=None):
        
        self.pos_bias_u = pos_bias_u
        self.pos_bias_v = pos_bias_v
        self.self_attention_model = self_attention_model
        self.n_heads = n_heads
        self.conv_kernel_size = conv_kernel_size
        self.dropout = dropout
        self.dropout_att = dropout_att
        self.conv_norm_type = conv_norm_type
        self.d_ff = d_ff
        self.d_model = d_model
        
        super(OverwrittenConformerLayer, self).__init__(
            self.d_model,
            self.d_ff,
            self_attention_model=self.self_attention_model,
            conv_kernel_size=self.conv_kernel_size,
            conv_norm_type=self.conv_norm_type,
            n_heads=self.n_heads,
            dropout=self.dropout,
            dropout_att=self.dropout_att,
            pos_bias_u=self.pos_bias_u,
            pos_bias_v=self.pos_bias_v,
        )
    
    def forward(self, x, att_mask=None, pos_emb=None, pad_mask=None):
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + self.dropout(x) * self.fc_factor
        
        x = self.norm_self_att(residual)
        if self.self_attention_model == 'rel_pos':
            x = self.self_attn(query=x, key=x, value=x, mask=att_mask, pos_emb=pos_emb)
        elif self.self_attention_model == 'abs_pos':
            x = self.self_attn(query=x, key=x, value=x, mask=att_mask)
        else:
            x = None
        self_attention_output = x.requires_grad_()
        residual = residual + self.dropout(x)
        
        x = self.norm_conv(residual)
        x = self.conv(x, pad_mask)
        residual = residual + self.dropout(x)
        
        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual = residual + self.dropout(x) * self.fc_factor
        
        x = self.norm_out(residual).requires_grad_()
        
        if self.is_adapter_available():
            # Call the adapters
            x = self.forward_enabled_adapters(x).requires_grad_()
        
        if self.is_access_enabled():
            self.register_accessible_tensor(tensor=x).requires_grad_()
        
        return x, self_attention_output