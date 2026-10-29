import copy
import io

import pytest
import torch

from src.optim import Adam, NAdam, SGD, lr_at


OPTIMIZERS = ("sgd", "adam", "nadam")


def make_optim(name, params, weight_decay=0.0):
    if name == "sgd":
        return SGD(params, lr=0.025, momentum=0.85, nesterov=True, weight_decay=weight_decay)
    if name == "adam":
        return Adam(params, lr=0.004, beta1=0.87, beta2=0.992, eps=1e-9, weight_decay=weight_decay)
    return NAdam(
        params, lr=0.004, beta1=0.87, beta2=0.992, eps=1e-9,
        momentum_decay=0.006, weight_decay=weight_decay,
    )


def make_reference(name, params, weight_decay=0.0):
    # Keep all torch.optim references in tests, never in src/.
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=0.025, momentum=0.85, nesterov=True, weight_decay=weight_decay
        )
    if name == "adam":
        return torch.optim.AdamW(
            params, lr=0.004, betas=(0.87, 0.992), eps=1e-9, weight_decay=weight_decay
        )
    return torch.optim.NAdam(
        params, lr=0.004, betas=(0.87, 0.992), eps=1e-9,
        momentum_decay=0.006, weight_decay=weight_decay,
        decoupled_weight_decay=True,
    )


@pytest.mark.parametrize("name", OPTIMIZERS)
@pytest.mark.parametrize("weight_decay", (0.0, 0.03))
def test_matches_torch_reference(name, weight_decay):
    torch.manual_seed(41)
    custom_model = torch.nn.Sequential(
        torch.nn.Linear(5, 9), torch.nn.Tanh(), torch.nn.Linear(9, 3)
    ).double()
    reference_model = copy.deepcopy(custom_model)
    custom = make_optim(name, custom_model.parameters(), weight_decay)
    reference = make_reference(name, reference_model.parameters(), weight_decay)
    x = torch.randn(13, 5, dtype=torch.float64)
    target = torch.randn(13, 3, dtype=torch.float64)

    for _ in range(200):
        custom.zero_grad()
        reference.zero_grad(set_to_none=True)
        torch.nn.functional.mse_loss(custom_model(x), target).backward()
        torch.nn.functional.mse_loss(reference_model(x), target).backward()
        custom.step()
        reference.step()

    max_diff = max(
        (left - right).abs().max().item()
        for left, right in zip(custom_model.parameters(), reference_model.parameters())
    )
    assert max_diff < 1e-8


@pytest.mark.parametrize("name", OPTIMIZERS)
def test_converges_on_convex_quadratic(name):
    param = torch.nn.Parameter(torch.tensor([8.0, -5.0, 2.5], dtype=torch.float64))
    target = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float64)
    optimizer = (
        SGD([param], lr=0.1, momentum=0.9)
        if name == "sgd"
        else Adam([param], lr=0.1)
        if name == "adam"
        else NAdam([param], lr=0.1)
    )
    for _ in range(1000):
        optimizer.zero_grad()
        ((param - target) ** 2).sum().backward()
        optimizer.step()
    assert ((param - target) ** 2).sum().item() < 1e-6


@pytest.mark.parametrize("name", OPTIMIZERS)
def test_checkpoint_resume_matches_uninterrupted(name):
    initial = torch.tensor([2.0, -1.0, 4.0], dtype=torch.float64)
    target = torch.tensor([-0.5, 3.0, 1.5], dtype=torch.float64)

    def run(param, optimizer, steps):
        for step in steps:
            optimizer.lr = lr_at(step, base_lr=0.02, warmup_steps=4, total_steps=100)
            optimizer.zero_grad()
            ((param - target) ** 2).sum().backward()
            optimizer.step()

    full_param = torch.nn.Parameter(initial.clone())
    full_opt = make_optim(name, [full_param])
    run(full_param, full_opt, range(100))

    first_param = torch.nn.Parameter(initial.clone())
    first_opt = make_optim(name, [first_param])
    run(first_param, first_opt, range(50))
    buffer = io.BytesIO()
    torch.save({"param": first_param.detach().clone(), "optimizer": first_opt.state_dict()}, buffer)
    buffer.seek(0)
    checkpoint = torch.load(buffer, weights_only=True)

    resumed_param = torch.nn.Parameter(checkpoint["param"].clone())
    resumed_opt = make_optim(name, [resumed_param])
    resumed_opt.load_state_dict(checkpoint["optimizer"])
    run(resumed_param, resumed_opt, range(50, 100))
    assert torch.equal(full_param, resumed_param)


def test_zero_grad_sets_grad_to_none():
    param = torch.nn.Parameter(torch.ones(2))
    param.grad = torch.ones_like(param)
    SGD([param], lr=0.1).zero_grad()
    assert param.grad is None


def test_lr_schedule_warmup_and_cosine_endpoints():
    args = {"base_lr": 1.0, "warmup_steps": 2, "total_steps": 102}
    assert lr_at(0, **args) == pytest.approx(0.5)
    assert lr_at(1, **args) == pytest.approx(1.0)
    assert lr_at(2, **args) == pytest.approx(1.0)
    assert lr_at(101, **args) == pytest.approx(0.01)
