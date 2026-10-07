import copy
import torch
from reptor import RepTorBlock


def _check(block: RepTorBlock, shape):
    block.train()
    with torch.no_grad():
        for _ in range(5):
            block(torch.randn(*shape))
    block.eval()
    x = torch.randn(*shape)
    with torch.no_grad():
        expected = block(x)
    deployed = copy.deepcopy(block).switch_to_deploy().eval()
    with torch.no_grad():
        actual = deployed(x)
    torch.testing.assert_close(expected, actual, rtol=1e-4, atol=1e-5)


def test_identity_branch_equivalence():
    torch.manual_seed(123)
    _check(RepTorBlock(24, 24, kmax=9, stride=1), (2, 24, 51))


def test_stride2_equivalence():
    torch.manual_seed(321)
    _check(RepTorBlock(16, 24, kmax=7, stride=2), (2, 16, 51))
