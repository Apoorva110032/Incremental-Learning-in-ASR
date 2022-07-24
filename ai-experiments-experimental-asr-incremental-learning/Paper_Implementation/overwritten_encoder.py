import math
from collections import OrderedDict
from typing import List, Optional

import torch
import torch.distributed
import torch.nn as nn
import os

from omegaconf import DictConfig, OmegaConf, open_dict
from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder
from nemo.collections.asr.parts.submodules.multi_head_attention import PositionalEncoding, RelPositionalEncoding
from nemo.collections.asr.parts.submodules.subsampling import ConvSubsampling, StackingSubsampling
from nemo.core.classes.common import typecheck
from nemo.core.classes.exportable import Exportable
from nemo.core.classes.module import NeuralModule
from nemo.core.neural_types import AcousticEncodedRepresentation, LengthsType, NeuralType, SpectrogramType
from overwritten_conformer_layer import OverwrittenConformerLayer

__all__ = ['OverwrittenEncoder']


class OverwrittenEncoder(ConformerEncoder, NeuralModule, Exportable):
    
    def __init__(self, feat_in, n_layers, d_model, feat_out=-1, subsampling='striding', subsampling_factor=4,
                 subsampling_conv_channels=-1,
                 ff_expansion_factor=4, self_attention_model='rel_pos', n_heads=8, att_context_size=None, xscaling=True,
                 untie_biases=True,
                 pos_emb_max_len=5000, conv_kernel_size=31, conv_norm_type='batch_norm', dropout=0.1, dropout_emb=0.1,
                 dropout_att=0.1):
        super().__init__(
            feat_in=feat_in,
            n_layers=n_layers,
            d_model=d_model,
            feat_out=-1,
            subsampling='striding',
            subsampling_factor=4,
            subsampling_conv_channels=-1,
            ff_expansion_factor=4,
            self_attention_model='rel_pos',
            n_heads=8,
            att_context_size=None,
            xscaling=True,
            untie_biases=True,
            pos_emb_max_len=5000,
            conv_kernel_size=31,
            conv_norm_type='batch_norm',
            dropout=0.1,
            dropout_emb=0.1,
            dropout_att=0.1,
        )
        
        d_ff = d_model * ff_expansion_factor
        self.d_model = d_model
        self._feat_in = feat_in
        self.scale = math.sqrt(self.d_model)
        if att_context_size:
            self.att_context_size = att_context_size
        else:
            self.att_context_size = [-1, -1]
        
        if xscaling:
            self.xscale = math.sqrt(d_model)
        else:
            self.xscale = None
        
        if subsampling_conv_channels == -1:
            subsampling_conv_channels = d_model
        if subsampling and subsampling_factor > 1:
            if subsampling == 'stacking':
                self.pre_encode = StackingSubsampling(
                    subsampling_factor=subsampling_factor, feat_in=feat_in, feat_out=d_model
                )
            else:
                self.pre_encode = ConvSubsampling(
                    subsampling=subsampling,
                    subsampling_factor=subsampling_factor,
                    feat_in=feat_in,
                    feat_out=d_model,
                    conv_channels=subsampling_conv_channels,
                    activation=nn.ReLU(),
                )
        else:
            self.pre_encode = nn.Linear(feat_in, d_model)
        
        self._feat_out = d_model
        
        if not untie_biases and self_attention_model == "rel_pos":
            d_head = d_model // n_heads
            pos_bias_u = nn.Parameter(torch.Tensor(n_heads, d_head))
            pos_bias_v = nn.Parameter(torch.Tensor(n_heads, d_head))
            nn.init.zeros_(pos_bias_u)
            nn.init.zeros_(pos_bias_v)
        else:
            pos_bias_u = None
            pos_bias_v = None
        
        self.pos_emb_max_len = pos_emb_max_len
        if self_attention_model == "rel_pos":
            self.pos_enc = RelPositionalEncoding(
                d_model=d_model,
                dropout_rate=dropout,
                max_len=pos_emb_max_len,
                xscale=self.xscale,
                dropout_rate_emb=dropout_emb,
            )
        elif self_attention_model == "abs_pos":
            pos_bias_u = None
            pos_bias_v = None
            self.pos_enc = PositionalEncoding(
                d_model=d_model, dropout_rate=dropout, max_len=pos_emb_max_len, xscale=self.xscale
            )
        else:
            raise ValueError(f"Not valid self_attention_model: '{self_attention_model}'!")
        
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = OverwrittenConformerLayer(
                d_model=d_model,
                d_ff=d_ff,
                self_attention_model=self_attention_model,
                n_heads=n_heads,
                conv_kernel_size=conv_kernel_size,
                conv_norm_type=conv_norm_type,
                dropout=dropout,
                dropout_att=dropout_att,
                pos_bias_u=pos_bias_u,
                pos_bias_v=pos_bias_v,
            )
            self.layers.append(layer)
        
        if feat_out > 0 and feat_out != self._feat_out:
            self.out_proj = nn.Linear(self._feat_out, feat_out)
            self._feat_out = feat_out
        else:
            self.out_proj = None
            self._feat_out = d_model
        self.set_max_audio_length(self.pos_emb_max_len)
        self.use_pad_mask = True
    
    @property
    def input_types(self):
        """Returns definitions of module input ports."""
        return OrderedDict(
            {
                "audio_signal": NeuralType(('B', 'D', 'T'), SpectrogramType()),
                "length": NeuralType(tuple('B'), LengthsType()),
            }
        )
    
    @property
    def output_types(self):
        """Returns definitions of module output ports."""
        return OrderedDict(
            {
                "outputs": NeuralType(('B', 'D', 'T'), AcousticEncodedRepresentation()),
                "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
                "self_attention_outputs": NeuralType(('B', 'T', 'D'), AcousticEncodedRepresentation())
            }
        )
    
    @typecheck()
    def forward_for_export(self, audio_signal, length):
        max_audio_length: int = audio_signal.size(-1)
        
        if max_audio_length > self.max_audio_length:
            self.set_max_audio_length(max_audio_length)
        
        if length is None:
            length = audio_signal.new_full(
                audio_signal.size(0), max_audio_length, dtype=torch.int32, device=self.seq_range.device
            )
        
        audio_signal = torch.transpose(audio_signal, 1, 2)
        
        if isinstance(self.pre_encode, nn.Linear):
            audio_signal = self.pre_encode(audio_signal)
        else:
            audio_signal, length = self.pre_encode(audio_signal, length)
        
        audio_signal, pos_emb = self.pos_enc(audio_signal)
        # adjust size
        max_audio_length = audio_signal.size(1)
        # Create the self-attention and padding masks
        
        pad_mask = self.make_pad_mask(max_audio_length, length)
        att_mask = pad_mask.unsqueeze(1).repeat([1, max_audio_length, 1])
        att_mask = torch.logical_and(att_mask, att_mask.transpose(1, 2))
        if self.att_context_size[0] >= 0:
            att_mask = att_mask.triu(diagonal=-self.att_context_size[0])
        if self.att_context_size[1] >= 0:
            att_mask = att_mask.tril(diagonal=self.att_context_size[1])
        att_mask = ~att_mask
        
        if self.use_pad_mask:
            pad_mask = ~pad_mask
        else:
            pad_mask = None
        
        for lth, layer in enumerate(self.layers):
            audio_signal, self_attention_outputs = layer(x=audio_signal, att_mask=att_mask, pos_emb=pos_emb,
                                                         pad_mask=pad_mask)
        
        if self.out_proj is not None:
            audio_signal = self.out_proj(audio_signal).requires_grad_()
        
        audio_signal = torch.transpose(audio_signal, 1, 2)
        return audio_signal, length, self_attention_outputs