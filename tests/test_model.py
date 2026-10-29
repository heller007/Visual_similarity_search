import torch

from src.model import BasicBlock, ResNet18


def test_resnet18_outputs_unit_embeddings():
    model = ResNet18(d=128).eval()
    with torch.no_grad():
        embeddings = model(torch.randn(2, 3, 32, 32))
    assert embeddings.shape == (2, 128)
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(2), atol=1e-5)
    assert 11_000_000 < sum(p.numel() for p in model.parameters()) < 11_300_000


def test_resnet18_supports_ablation_dimensions():
    x = torch.randn(2, 3, 32, 32)
    for d in (64, 256):
        with torch.no_grad():
            assert ResNet18(d=d).eval()(x).shape == (2, d)


def test_basic_block_preserves_and_downsamples_shapes():
    assert BasicBlock(64, 64)(torch.randn(2, 64, 16, 16)).shape == (2, 64, 16, 16)
    assert BasicBlock(64, 128, stride=2)(torch.randn(2, 64, 16, 16)).shape == (2, 128, 8, 8)
