import torch
from reptor import reptor9, reptor_a


def test_reptor9_shape():
    assert reptor9().eval()(torch.randn(2, 40, 101)).shape == (2, 12)


def test_reptor_a_shape():
    assert reptor_a().eval()(torch.randn(2, 40, 101)).shape == (2, 12)
