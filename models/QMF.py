"""Quantum-inspired Multimodal Fusion (QMF).

The original repository implemented this architecture under the historical
name ``LocalMixtureNN``.  This public class gives the paper model an explicit
and reproducible model name without duplicating its implementation.
"""

from .LocalMixtureNN import LocalMixtureNN


class QMF(LocalMixtureNN):
    """QMF model: complex tensor fusion followed by local n-gram mixtures."""

    pass
