"""
Gradient and activation diagnostic for SSM neck models.
Run this to examine whether SSM components are learning or disrupting gradients.

Usage:
    cd models/yolov9
    python diag_gradients.py --cfg models/detect/gelan-s-s4nd-neck.yaml --device 0
    python diag_gradients.py --cfg models/detect/gelan-s.yaml --device 0  # baseline
"""
import argparse
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.yolo import Model


def grad_norm(params):
    total = 0.0
    n = 0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().norm().item() ** 2
            n += 1
    return (total ** 0.5) if n > 0 else 0.0, n


def param_norm(params):
    total = sum(p.detach().norm().item() ** 2 for p in params)
    return total ** 0.5


def run(cfg, device):
    dev = torch.device(f'cuda:{device}')
    model = Model(cfg, ch=3, nc=80).to(dev)
    model.train()

    # Dummy forward + backward
    x = torch.randn(2, 3, 640, 640, device=dev)
    dummy_target = []
    out = model(x)

    # Use sum of all outputs as a proxy loss
    if isinstance(out, (list, tuple)):
        loss = sum(o.float().sum() for o in out if isinstance(o, torch.Tensor))
    else:
        loss = out.float().sum()
    loss.backward()

    # ── Parameter group separation ───────────────────────────────────────────
    backbone, fpn_pan, ssm, skip, other = [], [], [], [], []

    for name, param in model.named_parameters():
        n = name.lower()
        if any(f'model.{i}.' in name for i in range(9)):
            backbone.append((name, param))
        elif 's4nd_neck' in n or 'mamband_neck' in n:
            ssm.append((name, param))
        elif 'skip_proj' in n:
            skip.append((name, param))
        elif any(f'model.{i}.' in name for i in range(9, 22)):
            fpn_pan.append((name, param))
        else:
            other.append((name, param))

    print("\n" + "=" * 70)
    print(f"GRADIENT DIAGNOSTICS  cfg={cfg}")
    print("=" * 70)

    groups = [
        ("Backbone (layers 0-8)", backbone),
        ("FPN/PAN  (layers 9-21)", fpn_pan),
        ("SSM neck (s4nd/mamband)", ssm),
        ("Skip proj", skip),
    ]

    for label, group in groups:
        params = [p for _, p in group]
        gnorm, n_with_grad = grad_norm(params)
        pnorm = param_norm(params)
        n_total = len(params)
        print(f"\n{label}")
        print(f"  params:          {n_total} tensors, {sum(p.numel() for p in params):,} elements")
        print(f"  param norm:      {pnorm:.4f}")
        print(f"  grad norm:       {gnorm:.4f}  ({n_with_grad}/{n_total} tensors have grad)")
        if pnorm > 0:
            print(f"  grad/param:      {gnorm / (pnorm + 1e-8):.4f}  (relative gradient magnitude)")

    # ── SSM output magnitude (forward hook) ─────────────────────────────────
    print("\n" + "-" * 70)
    print("SSM FORWARD OUTPUT MAGNITUDES (fresh forward pass with hooks)")
    print("-" * 70)

    handles = []
    output_stats = {}

    def make_hook(tag):
        def hook(module, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            output_stats[tag] = {
                'mean_abs': t.detach().abs().mean().item(),
                'std':      t.detach().std().item(),
                'max_abs':  t.detach().abs().max().item(),
            }
        return hook

    for name, module in model.named_modules():
        n = name.lower()
        if 's4nd_neck' in n and hasattr(module, 'decoder'):
            handles.append(module.register_forward_hook(make_hook('s4nd_neck_internal')))
        if 'mamband_neck' in n and hasattr(module, 'mamba'):
            handles.append(module.register_forward_hook(make_hook('mamband_neck_internal')))
        if 'skip_proj' in n and isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(make_hook('skip_proj')))

    with torch.no_grad():
        model(x)

    for h in handles:
        h.remove()

    if output_stats:
        for tag, stats in output_stats.items():
            print(f"\n  {tag}:")
            print(f"    mean |x|: {stats['mean_abs']:.4f}")
            print(f"    std:      {stats['std']:.4f}")
            print(f"    max |x|:  {stats['max_abs']:.4f}")
    else:
        print("  (no SSM modules found — this is a baseline model)")

    print("\n" + "=" * 70)
    print("HOW TO READ THESE RESULTS:")
    print("  grad/param ratio:")
    print("    ~0.01-0.1  = healthy learning signal")
    print("    <0.001     = vanishing gradients, SSM not learning")
    print("    >1.0       = exploding gradients, SSM destabilizing")
    print("  SSM mean |x| vs skip_proj mean |x|:")
    print("    SSM << skip  = SSM not contributing (zero-init working but stuck)")
    print("    SSM ~= skip  = SSM contributing equally")
    print("    SSM >> skip  = SSM dominating, could destabilize")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, required=True)
    parser.add_argument('--device', type=int, default=0)
    opt = parser.parse_args()
    run(opt.cfg, opt.device)
