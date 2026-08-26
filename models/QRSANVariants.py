"""Controlled real-valued and real/imaginary-readout QRSAN baselines."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .QDNN_ATTENTION import LSTMSubNet, MLPSubNet, uQDNN_ATTENTION


class RealProjectionMeasurement(nn.Module):
    """Real-valued analogue of QRSAN's K learned projection directions."""

    def __init__(self, dimension: int, units: int):
        super().__init__()
        self.directions = nn.Parameter(torch.empty(units, dimension))
        nn.init.xavier_uniform_(self.directions)

    def forward(self, values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        directions = F.normalize(self.directions, p=2, dim=1, eps=1e-10)
        scores = torch.matmul(values, directions.t()).square()
        return torch.matmul(weights.transpose(-1, -2), scores).squeeze(1)


class RealQRSAN(nn.Module):
    """Parameter-matchable real-valued QRSAN control.

    It shares QRSAN's input projections, tensor-product interaction dimension,
    temporal attention, residual path and prediction MLP.  It intentionally has
    no phase embedding, imaginary representation, complex multiplication or
    complex measurement layer.
    """

    def __init__(self, opt):
        super().__init__()
        self.device = opt.device
        self.input_dims = opt.input_dims
        self.feature_indexes = opt.feature_indexes
        self.text_hidden_dim = opt.text_hidden_dim
        self.output_dim = opt.output_dim
        self.num_measurements = opt.measurement_size
        self.output_cell_dim = opt.output_cell_dim
        self.residual_self_attention = getattr(opt, "residual_self_attention", True)
        self.subnet_dropout_rates = [float(x) for x in str(opt.subnet_dropout_rates).split(",")]
        self.contracted_dims = [int(x) for x in str(opt.contracted_dims).split(",")]
        if len(self.subnet_dropout_rates) == 1:
            self.subnet_dropout_rates *= len(self.input_dims)
        if len(self.contracted_dims) == 1:
            self.contracted_dims *= len(self.input_dims)
        if len(self.contracted_dims) != len(self.input_dims):
            raise ValueError("contracted_dims must provide one dimension per modality")
        embedding = torch.tensor(opt.lookup_table, dtype=torch.float)
        self.embed = nn.Embedding.from_pretrained(embedding, freeze=opt.amplitude_freeze)
        self.proj_layers = nn.ModuleList()
        for index, input_dim in enumerate(self.input_dims):
            if index == 0:
                self.proj_layers.append(LSTMSubNet(input_dim, self.text_hidden_dim, self.contracted_dims[index], self.subnet_dropout_rates[index], self.device))
            else:
                self.proj_layers.append(MLPSubNet(input_dim, self.contracted_dims[index], self.subnet_dropout_rates[index]))
        self.modality_weights = nn.Parameter(torch.zeros(len(self.input_dims)))
        self.measurement_dim = 1
        for dim in self.contracted_dims:
            self.measurement_dim *= dim
        self.real_softmax = nn.Softmax(dim=2)
        self.measurement = RealProjectionMeasurement(self.measurement_dim, self.num_measurements)
        self.fc_out = nn.Sequential(
            nn.Dropout(opt.output_dropout_rate), nn.Linear(self.num_measurements, self.output_cell_dim), nn.ReLU(),
            nn.Linear(self.output_cell_dim, self.output_cell_dim), nn.ReLU(), nn.Linear(self.output_cell_dim, self.output_dim),
        )

    def forward(self, in_modalities):
        in_modalities = [self.embed(x) if x.ndim == 2 else x for x in in_modalities]
        projected = [F.normalize(layer(value), p=2, dim=-1, eps=1e-10) for layer, value in zip(self.proj_layers, in_modalities)]
        norms = [torch.softmax(value.norm(dim=-1, keepdim=True), dim=1) for value in projected]
        modality_weights = torch.softmax(self.modality_weights, dim=0)
        temporal_weights = sum(weight * norm for weight, norm in zip(modality_weights, norms))
        products = []
        for time_index in range(projected[0].shape[1]):
            product = torch.ones(projected[0].shape[0], 1, device=projected[0].device)
            for value in projected:
                product = torch.bmm(product.unsqueeze(2), value[:, time_index, :].unsqueeze(1)).flatten(1)
            products.append(product)
        fused = torch.stack(products, dim=1)
        attended = torch.bmm(self.real_softmax(torch.bmm(fused, fused.transpose(1, 2))), fused)
        if self.residual_self_attention:
            attended = attended + fused
        return self.fc_out(self.measurement(attended, temporal_weights))


class RealImagConcatMLP(uQDNN_ATTENTION):
    """QRSAN representation with conventional [real; imaginary] MLP readout."""

    def __init__(self, opt):
        opt.readout_mode = "real_imag_concat_mlp"
        super().__init__(opt)
