# -*- coding: utf-8 -*-
"""MEGAKANs adapted for aligned, pre-extracted multimodal sequences.

Keeps the paper's multi-granularity encoders, multi-dilated visual feature
aggregation, global channel-spatial attention and KAN fusion/prediction.
"""
import torch
from torch import nn

from .modern_layers import AttentivePool, InputEmbedding, KAN, opt_value


class TextMultiGranularityEncoder(nn.Module):
    def __init__(self, input_dim, dim, heads, dropout):
        super(TextMultiGranularityEncoder, self).__init__()
        branch_dim = max(dim // 3, 1)
        self.local = nn.ModuleList([
            nn.Conv1d(input_dim, branch_dim, kernel_size=k, padding=k // 2)
            for k in (1, 3, 5)
        ])
        local_dim = branch_dim * 3
        self.local_proj = nn.Linear(local_dim, dim)
        gru_hidden = max(dim // 2, 1)
        self.global_gru = nn.GRU(input_dim, gru_hidden, batch_first=True,
                                 bidirectional=True)
        self.global_proj = nn.Linear(gru_hidden * 2, dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.pool = AttentivePool(dim)

    def forward(self, x):
        conv_input = x.transpose(1, 2)
        local = torch.cat([torch.relu(conv(conv_input)) for conv in self.local], dim=1)
        local = self.local_proj(local.transpose(1, 2))
        global_feature, _ = self.global_gru(x)
        global_feature = self.global_proj(global_feature)
        query = local.transpose(0, 1)
        context = global_feature.transpose(0, 1)
        attended, _ = self.attention(query, context, context, need_weights=False)
        return self.pool(self.norm(local + attended.transpose(0, 1) + global_feature))


class VisualMDFA(nn.Module):
    def __init__(self, input_dim, dim):
        super(VisualMDFA, self).__init__()
        self.input_proj = nn.Conv1d(input_dim, dim, kernel_size=1)
        self.dilated = nn.ModuleList([
            nn.Conv1d(dim, dim, kernel_size=3, padding=d, dilation=d)
            for d in (1, 3, 6)
        ])
        self.bottleneck = nn.Conv1d(dim * 4, dim, kernel_size=1)
        self.pool = AttentivePool(dim)

    def forward(self, x):
        base = torch.relu(self.input_proj(x.transpose(1, 2)))
        scales = [base] + [torch.relu(conv(base)) for conv in self.dilated]
        feature = torch.relu(self.bottleneck(torch.cat(scales, dim=1)))
        return self.pool(feature.transpose(1, 2))


class AcousticEncoder(nn.Module):
    def __init__(self, input_dim, dim):
        super(AcousticEncoder, self).__init__()
        hidden = max(dim // 2, 1)
        self.gru = nn.GRU(input_dim, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden * 2, dim)
        self.pool = AttentivePool(dim)

    def forward(self, x):
        x, _ = self.gru(x)
        return self.pool(self.proj(x))


class GlobalChannelSpatialAttention(nn.Module):
    def __init__(self, dim, grid_size, dropout):
        super(GlobalChannelSpatialAttention, self).__init__()
        hidden = max(dim // 4, 8)
        self.channel = KAN([dim, hidden, dim], grid_size, dropout)
        self.spatial = nn.Conv1d(2, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # x: [batch, modality/channel, latent feature/spatial]
        channel_summary = x.mean(dim=1)
        channel_gate = torch.sigmoid(self.channel(channel_summary)).unsqueeze(1)
        x = x * channel_gate
        spatial_summary = torch.stack([x.mean(dim=2), x.max(dim=2)[0]], dim=1)
        spatial_gate = torch.sigmoid(self.spatial(spatial_summary)).transpose(1, 2)
        return x * spatial_gate


class MEGAKANs(nn.Module):
    def __init__(self, opt):
        super(MEGAKANs, self).__init__()
        if len(opt.input_dims) != 3:
            raise ValueError('MEGAKANs requires textual, visual and acoustic modalities')
        self.input_embed = InputEmbedding(opt)
        dim = int(opt_value(opt, 'megakan_dim', 96))
        heads = int(opt_value(opt, 'megakan_heads', 4))
        dropout = float(opt_value(opt, 'megakan_dropout', 0.1))
        grid_size = int(opt_value(opt, 'megakan_grid_size', 8))
        if dim % heads != 0:
            raise ValueError('megakan_dim must be divisible by megakan_heads')

        text_dim, visual_dim, audio_dim = opt.input_dims
        self.text_encoder = TextMultiGranularityEncoder(text_dim, dim, heads, dropout)
        self.visual_encoder = VisualMDFA(visual_dim, dim)
        self.audio_encoder = AcousticEncoder(audio_dim, dim)
        self.alignment = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU())
            for _ in range(3)
        ])
        self.gcsa = GlobalChannelSpatialAttention(dim, grid_size, dropout)
        fusion_hidden = int(opt_value(opt, 'megakan_fusion_dim', dim * 2))
        self.fusion = KAN([dim * 3, fusion_hidden, dim], grid_size, dropout)
        self.output = KAN([dim, max(dim // 2, 8), opt.output_dim],
                          grid_size, dropout)

    def forward(self, in_modalities):
        text, visual, audio = self.input_embed(in_modalities)
        features = [self.text_encoder(text), self.visual_encoder(visual),
                    self.audio_encoder(audio)]
        aligned = torch.stack([
            layer(feature) for layer, feature in zip(self.alignment, features)
        ], dim=1)
        refined = self.gcsa(aligned).reshape(aligned.size(0), -1)
        return self.output(self.fusion(refined))
