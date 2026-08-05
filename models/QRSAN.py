"""Quantum Residual Self-Attention Network (QRSAN).

The architecture was originally exposed as ``uQDNN_ATTENTION``.  QRSAN is a
clear paper-facing alias for the same quantum self-attention and residual
measurement network.
"""

from .QDNN_ATTENTION import uQDNN_ATTENTION


class QRSAN(uQDNN_ATTENTION):
    """Quantum self-attention network with a residual connection."""

    pass
