"""Simple tensor-only optimizers."""

from copy import deepcopy
from math import cos, pi

import torch


def lr_at(step, *, base_lr, warmup_steps, total_steps):
    """Linear warmup, then cosine decay to 1% of base_lr."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps - 1)
    return base_lr * (0.01 + 0.99 * (1 + cos(pi * progress)) / 2)


class Optimizer:
    options = ()

    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr
        self.state = [{} for _ in self.params]

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def state_dict(self):
        return {
            "lr": self.lr,
            "options": {key: getattr(self, key) for key in self.options},
            "state": deepcopy(self.state),
        }

    def load_state_dict(self, saved):
        if len(saved["state"]) != len(self.params):
            raise ValueError("Checkpoint parameter count does not match.")
        self.lr = saved["lr"]
        for key, value in saved["options"].items():
            setattr(self, key, value)
        self.state = []
        for param, saved_slot in zip(self.params, saved["state"]):
            slot = {}
            for key, value in saved_slot.items():
                if torch.is_tensor(value):
                    slot[key] = value.to(param) if value.is_floating_point() else value.to(param.device)
                else:
                    slot[key] = deepcopy(value)
            self.state.append(slot)


class SGD(Optimizer):
    """SGD with coupled L2 weight decay and optional momentum."""

    options = ("momentum", "nesterov", "weight_decay")

    def __init__(self, params, lr=0.01, momentum=0.0, nesterov=False, weight_decay=0.0):
        super().__init__(params, lr)
        self.momentum, self.nesterov, self.weight_decay = momentum, nesterov, weight_decay

    @torch.no_grad()
    def step(self):
        for p, state in zip(self.params, self.state):
            if p.grad is None:
                continue
            grad = p.grad.add(p, alpha=self.weight_decay) if self.weight_decay else p.grad
            if self.momentum:
                if "buffer" not in state:
                    state["buffer"] = grad.clone()
                else:
                    state["buffer"].mul_(self.momentum).add_(grad)
                grad = grad.add(state["buffer"], alpha=self.momentum) if self.nesterov else state["buffer"]
            p.add_(grad, alpha=-self.lr)


class Adam(Optimizer):
    """Bias-corrected Adam with decoupled AdamW-style weight decay."""

    options = ("beta1", "beta2", "eps", "weight_decay")

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr)
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.weight_decay = weight_decay

    @torch.no_grad()
    def step(self):
        for p, state in zip(self.params, self.state):
            if p.grad is None:
                continue
            if not state:
                state.update(step=0, m=torch.zeros_like(p), v=torch.zeros_like(p))
            state["step"] += 1
            if self.weight_decay:
                p.mul_(1 - self.lr * self.weight_decay)
            g = p.grad
            state["m"].mul_(self.beta1).add_(g, alpha=1 - self.beta1)
            state["v"].mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
            t = state["step"]
            denom = (state["v"] / (1 - self.beta2**t)).sqrt().add_(self.eps)
            p.addcdiv_(state["m"], denom, value=-self.lr / (1 - self.beta1**t))


class NAdam(Optimizer):
    """PyTorch-style NAdam (Dozat, 2016) with decoupled weight decay."""

    options = ("beta1", "beta2", "eps", "momentum_decay", "weight_decay")

    def __init__(
        self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
        momentum_decay=0.004, weight_decay=0.0,
    ):
        super().__init__(params, lr)
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.momentum_decay, self.weight_decay = momentum_decay, weight_decay

    @torch.no_grad()
    def step(self):
        for p, state in zip(self.params, self.state):
            if p.grad is None:
                continue
            if not state:
                state.update(step=0, mu_product=1.0, m=torch.zeros_like(p), v=torch.zeros_like(p))
            state["step"] += 1
            t = state["step"]
            if self.weight_decay:
                p.mul_(1 - self.lr * self.weight_decay)
            g = p.grad
            mu = self.beta1 * (1 - 0.5 * 0.96 ** (t * self.momentum_decay))
            mu_next = self.beta1 * (1 - 0.5 * 0.96 ** ((t + 1) * self.momentum_decay))
            state["mu_product"] *= mu
            state["m"].mul_(self.beta1).add_(g, alpha=1 - self.beta1)
            state["v"].mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
            denom = (state["v"] / (1 - self.beta2**t)).sqrt().add_(self.eps)
            next_product = state["mu_product"] * mu_next
            p.addcdiv_(g, denom, value=-self.lr * (1 - mu) / (1 - state["mu_product"]))
            p.addcdiv_(state["m"], denom, value=-self.lr * mu_next / (1 - next_product))
