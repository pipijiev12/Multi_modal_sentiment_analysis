# -*- coding: utf-8 -*-
"""Adaptive Language-guided Multimodal Transformer (ALMT).

Adapted to this repository's pre-extracted [text, vision, audio] sequences.
The AHL and language-to-hyper-modality fusion follow Zhang et al. (EMNLP 2023).
"""
import torch
from torch import nn

from .modern_layers import (CrossAttentionBlock, InputEmbedding,
                            SelfAttentionBlock, TokenProjector, opt_value)


class AdaptiveHyperModalityLayer(nn.Module):
    def __init__(self, dim, heads, dropout):
        super(AdaptiveHyperModalityLayer, self).__init__()
        self.language_audio = CrossAttentionBlock(dim, heads, dropout)
        self.language_visual = CrossAttentionBlock(dim, heads, dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, hyper, language, audio, visual):
        # Language is the query used to select useful non-verbal information.
        selected_audio = self.language_audio(language, audio) - language
        selected_visual = self.language_visual(language, visual) - language
        return self.norm(hyper + selected_audio + selected_visual)


class ALMT(nn.Module):
    def __init__(self, opt):
        super(ALMT, self).__init__()
        if len(opt.input_dims) != 3:
            raise ValueError('ALMT requires textual, visual and acoustic modalities')
        self.input_embed = InputEmbedding(opt)
        dim = int(opt_value(opt, 'almt_dim', 128))
        token_len = int(opt_value(opt, 'almt_token_len', 8))
        heads = int(opt_value(opt, 'almt_heads', 8))
        depth = int(opt_value(opt, 'almt_depth', 3))
        dropout = float(opt_value(opt, 'almt_dropout', 0.1))

        # Repository modality order is text, vision, audio.
        text_dim, visual_dim, audio_dim = opt.input_dims
        self.text_projector = TokenProjector(text_dim, dim, token_len, heads, dropout)
        self.visual_projector = TokenProjector(visual_dim, dim, token_len, heads, dropout)
        self.audio_projector = TokenProjector(audio_dim, dim, token_len, heads, dropout)

        self.language_layers = nn.ModuleList([
            SelfAttentionBlock(dim, heads, dropout) for _ in range(max(depth - 1, 0))
        ])
        self.ahl_layers = nn.ModuleList([
            AdaptiveHyperModalityLayer(dim, heads, dropout) for _ in range(depth)
        ])
        self.hyper_token = nn.Parameter(torch.ones(1, token_len, dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.fusion = CrossAttentionBlock(dim, heads, dropout)
        self.output = nn.Sequential(nn.LayerNorm(dim), nn.Dropout(dropout),
                                    nn.Linear(dim, opt.output_dim))

    def forward(self, in_modalities):
        text, visual, audio = self.input_embed(in_modalities)
        language = self.text_projector(text)
        visual = self.visual_projector(visual)
        audio = self.audio_projector(audio)

        language_scales = [language]
        for layer in self.language_layers:
            language_scales.append(layer(language_scales[-1]))

        hyper = self.hyper_token.expand(text.size(0), -1, -1)
        for index, layer in enumerate(self.ahl_layers):
            scale = language_scales[min(index, len(language_scales) - 1)]
            hyper = layer(hyper, scale, audio, visual)

        cls = self.cls_token.expand(text.size(0), -1, -1)
        language_query = torch.cat([cls, language_scales[-1]], dim=1)
        fused = self.fusion(language_query, hyper)
        return self.output(fused[:, 0])
