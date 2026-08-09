"""Quantum Residual Self-Attention Network (QRSAN).

The architecture was originally exposed as ``uQDNN_ATTENTION``.  QRSAN is a
clear paper-facing alias for the same quantum self-attention and residual
measurement network.
"""

from .QDNN_ATTENTION import uQDNN_ATTENTION


class QRSAN(uQDNN_ATTENTION):
    """Quantum self-attention network with a residual connection."""

    pass


class QSAN(uQDNN_ATTENTION):
    """Quantum Self-Attention Network: QDNN replaced by self-attention."""

    def __init__(self, opt):
        opt.residual_self_attention = False
        super().__init__(opt)


# Keep existing experiment configurations reproducible after adopting the
# paper-facing QSAN name for this ablation.
QRSANNoResidual = QSAN
