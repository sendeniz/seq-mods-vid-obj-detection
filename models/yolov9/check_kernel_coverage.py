"""
Compare spatial coverage: Baseline SPPELAN vs S4ND PE bidirectional kernel.
Impulse response at center → shows which spatial positions each module aggregates from.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from models.yolo import Model
from models.s4nd_model import S4NDNeckConcatPE


def get_s4nd_kernel_response(module, img_size, device='cpu'):
    """Impulse response of S4NDNeckConcatPE in isolation."""
    module.eval()
    H, W = img_size, img_size
    c1 = module.pos_embed.shape[1]
    x = torch.zeros(1, c1, H, W, device=device)
    x[0, 0, H//2, W//2] = 1.0
    with torch.no_grad():
        s4nd_out, _ = module.s4nd_neck(x, t=None, state=None)
    return s4nd_out[0].mean(0).cpu().numpy()


def get_sppelan_response(module, c1, img_size, device='cpu'):
    """Impulse response of SPPELAN in isolation."""
    module.eval()
    H, W = img_size, img_size
    x = torch.zeros(1, c1, H, W, device=device)
    x[0, 0, H//2, W//2] = 1.0
    with torch.no_grad():
        out = module._forward(x)
    return out[0].mean(0).cpu().numpy()


def print_quadrant_stats(response, img_size, name):
    cy, cx = img_size // 2, img_size // 2
    tl = response[:cy, :cx]
    tr = response[:cy, cx+1:]
    bl = response[cy+1:, :cx]
    br = response[cy+1:, cx+1:]
    print(f"  {name}:")
    print(f"    Top-left: {np.abs(tl).mean():.4f}  Top-right: {np.abs(tr).mean():.4f}")
    print(f"    Bot-left: {np.abs(bl).mean():.4f}  Bot-right: {np.abs(br).mean():.4f}")
    ratio = max(np.abs(tl).mean(), np.abs(tr).mean(), np.abs(bl).mean(), np.abs(br).mean()) / \
            (min(np.abs(tl).mean(), np.abs(tr).mean(), np.abs(bl).mean(), np.abs(br).mean()) + 1e-8)
    print(f"    Max/min quadrant ratio: {ratio:.2f}x")


def main():
    device = 'cpu'

    # --- Load S4ND PE model ---
    print("Loading S4ND PE checkpoint...")
    s4nd_ckpt = torch.load('train/gelan-s-cocosmall-pe/weights/best.pt', map_location=device)
    s4nd_model = Model('models/detect/gelan-s-ssm-concat-s4nd-pe.yaml', ch=3, nc=80)
    state = {k: v for k, v in s4nd_ckpt['model'].items() if 'anchor' not in k}
    s4nd_model.load_state_dict(state, strict=False)
    s4nd_model.eval()

    # --- Load Baseline model ---
    print("Loading baseline checkpoint...")
    base_ckpt = torch.load('train/gelan-s-cocosmall/weights/best.pt', map_location=device)
    base_model = Model('models/detect/gelan-s.yaml', ch=3, nc=80)
    state = {k: v for k, v in base_ckpt['model'].items() if 'anchor' not in k}
    base_model.load_state_dict(state, strict=False)
    base_model.eval()

    # Extract SPPELAN from baseline (layer 9, c1=256, c2=256, c3=128)
    sppelan = base_model.model[9]

    # Extract S4ND modules
    pe_modules = [(name, m) for name, m in s4nd_model.named_modules()
                  if isinstance(m, S4NDNeckConcatPE)]
    print(f"Found {len(pe_modules)} S4NDNeckConcatPE modules")

    # --- Compute responses ---
    print("\nBaseline SPPELAN @ P5 (20x20):")
    sppelan_response = get_sppelan_response(sppelan, c1=256, img_size=20, device=device)
    print_quadrant_stats(sppelan_response, 20, "SPPELAN P5")

    print("\nS4ND PE bidirectional:")
    s4nd_responses = []
    scale_names = ['P3 (80x80)', 'P4 (40x40)', 'P5 (20x20)']
    for (name, module), scale in zip(pe_modules, scale_names):
        img_size = module.pos_embed.shape[-1]
        response = get_s4nd_kernel_response(module, img_size=img_size, device=device)
        s4nd_responses.append((response, img_size, scale))
        print_quadrant_stats(response, img_size, f"S4ND {scale}")

    # --- Plot ---
    # Layout: [SPPELAN P5] [S4ND P3] [S4ND P4] [S4ND P5]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle('Spatial Coverage: Impulse Response at Center\n(mean abs response across channels)', fontsize=12)

    def plot_response(ax, response, title, img_size):
        cy, cx = img_size // 2, img_size // 2
        vmax = np.abs(response).max()
        im = ax.imshow(np.abs(response), cmap='hot', origin='upper', vmin=0, vmax=vmax)
        ax.axhline(cy, color='cyan', linewidth=0.8, linestyle='--', alpha=0.7)
        ax.axvline(cx, color='cyan', linewidth=0.8, linestyle='--', alpha=0.7)
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Quadrant annotations
        tl = np.abs(response[:cy, :cx]).mean()
        tr = np.abs(response[:cy, cx+1:]).mean()
        bl = np.abs(response[cy+1:, :cx]).mean()
        br = np.abs(response[cy+1:, cx+1:]).mean()
        ax.text(cx*0.25, cy*0.25, f'{tl:.4f}', color='white', fontsize=7, ha='center', va='center')
        ax.text(cx*1.75, cy*0.25, f'{tr:.4f}', color='white', fontsize=7, ha='center', va='center')
        ax.text(cx*0.25, cy*1.75, f'{bl:.4f}', color='white', fontsize=7, ha='center', va='center')
        ax.text(cx*1.75, cy*1.75, f'{br:.4f}', color='white', fontsize=7, ha='center', va='center')

    plot_response(axes[0], sppelan_response, 'Baseline SPPELAN\nP5 (20×20)', 20)

    for ax, (response, img_size, scale) in zip(axes[1:], s4nd_responses):
        plot_response(ax, response, f'S4ND PE (bidir)\n{scale}', img_size)

    plt.tight_layout()
    plt.savefig('kernel_coverage.png', dpi=150, bbox_inches='tight')
    print("\nSaved kernel_coverage.png")


if __name__ == '__main__':
    main()
