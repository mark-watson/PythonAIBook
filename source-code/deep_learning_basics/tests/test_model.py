"""Smoke tests for the CancerNet model.

Skips the CSV-loading path (which depends on ../machine-learning/) and just
exercises the model definition on synthetic input.
"""

import torch

from cancer_model import CancerNet


def test_cancer_net_forward_shape() -> None:
    model = CancerNet()
    model.eval()
    x = torch.zeros((4, 9), dtype=torch.float32)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (4, 1)


def test_cancer_net_is_trainable() -> None:
    model = CancerNet()
    params = list(model.parameters())
    assert len(params) > 0
    assert all(p.requires_grad for p in params)
