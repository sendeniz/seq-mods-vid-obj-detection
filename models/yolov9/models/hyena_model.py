"""
Hyena-2D neck module for YOLOv9.

Hyena (Poli et al. 2023) parameterizes convolution kernels implicitly via an FFN
applied to positional encodings, instead of deriving them from an SSM state matrix
like S4ND does. The key advantages over S4ND for detection:

  1. Full-rank 2D kernel -- S4ND uses K_2D = K_h x K_w (rank-1 outer product).
     Hyena learns K(i,j) directly via FFN(pos(i,j)), capturing cross-axis patterns
     that S4ND cannot represent with a single rank-1 term.

  2. Explicit 2D positional encoding -- the kernel knows its spatial coordinates
     (i/H, j/W) directly, which is well motivated for detection where spatial
     location matters for anchor priors and object localization.

  3. FFT-based -- the kernel is fixed (LTI), so the long-range convolution is
     computed efficiently via 2D FFT. When gating is disabled this is identical
     to S4ND's FFT path, just with a full-rank kernel instead of a rank-1 one.

  4. Optional multiplicative gating -- following the original Hyena paper, we can
     add input-dependent gating around the FFT conv. When bidirectional is also
     enabled we use per-direction gates so each kernel gets its own input mask.

  5. Much more parameter efficient -- the implicit FFN parameterization decouples
     parameter count from kernel size, so a small FFN generates a full H x W kernel.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SineActivation(nn.Module):
    # sine activation with learnable frequency -- same trick as Hyena paper
    # forces the FFN to learn high-frequency spatial patterns in the kernel
    def __init__(self, dim, w=10, train_freq=True):
        super().__init__()
        self.freq = nn.Parameter(w * torch.ones(1, dim)) if train_freq else w * torch.ones(1, dim)

    def forward(self, x):
        return torch.sin(self.freq * x)


def fftconv2d(u, h, D=None):
    """
    2D FFT convolution -- same idea as S4ND's FFT convolution but in 2D directly.
    Zero-pads to 2x size to avoid circular convolution artifacts at the edges.

    u: (B, C, H, W)  input feature map
    h: (C, H, W)     convolution kernel (one per channel)
    D: (C,) or None  optional learnable skip -- same as SSM's D term: y = conv(u,h) + D*u
                     both S4ND and Hyena have this, they just call it different things
                     if None (default) behavior is identical to before, nothing breaks
    returns: (B, C, H, W)
    """
    B, C, H, W = u.shape
    # pad to 2x to make it a linear convolution not circular
    fft_h, fft_w = 2 * H, 2 * W

    U_f = torch.fft.rfft2(u, s=(fft_h, fft_w))           # (B, C, fft_h, fft_w//2+1)
    H_f = torch.fft.rfft2(h, s=(fft_h, fft_w))           # (C, fft_h, fft_w//2+1)

    # broadcast over batch dim
    Y_f = U_f * H_f.unsqueeze(0)                          # (B, C, fft_h, fft_w//2+1)
    y = torch.fft.irfft2(Y_f, s=(fft_h, fft_w))          # (B, C, fft_h, fft_w)

    # crop back to original spatial size
    y = y[..., :H, :W]

    # D skip: direct feedthrough from input to output, same as SSM y = Cx + Du
    # zero-init means this contributes nothing at the start of training
    # to disable: just don't pass D (or pass None)
    if D is not None:
        y = y + D.view(1, C, 1, 1) * u

    return y


class Hyena2DFilter(nn.Module):
    """
    Generates a full-rank 2D convolution kernel via FFN on 2D positional encodings.

    The kernel h(i,j) is computed as:
        z(i,j) = [i/H, j/W, cos(f1*i), sin(f1*i), cos(f1*j), sin(f1*j), ...]
        h(i,j) = FFN(z(i,j)) * exp(-decay * dist(i,j))

    where FFN uses sine activations to capture high-frequency spatial patterns,
    and the exponential decay forces the kernel to have compact support.

    This gives a full-rank kernel -- no outer product factorization needed.
    """
    def __init__(self, d_model, img_size, bands=1, filter_order=16, kernel_layers=2, w=1, learnable_pe=False):
        super().__init__()
        self.d_model = d_model
        self.img_size = img_size
        self.learnable_pe = learnable_pe

        # positional encoding: [i, j] + [cos, sin] for each frequency band, per axis
        # emb_dim = 2 (coords) + 4*bands (sin/cos per axis per band)
        emb_dim = 2 + 4 * bands

        # build the 2D positional encoding grid
        H = W = img_size
        i_coord = torch.linspace(0, 1, H)
        j_coord = torch.linspace(0, 1, W)
        grid_i, grid_j = torch.meshgrid(i_coord, j_coord, indexing='ij')  # (H, W)

        freqs = torch.linspace(1, bands, bands) * 2 * math.pi  # frequency bands

        parts = [grid_i.unsqueeze(-1), grid_j.unsqueeze(-1)]   # raw position coords
        for f in freqs.tolist():
            parts += [
                torch.cos(f * grid_i).unsqueeze(-1),
                torch.sin(f * grid_i).unsqueeze(-1),
                torch.cos(f * grid_j).unsqueeze(-1),
                torch.sin(f * grid_j).unsqueeze(-1),
            ]
        z = torch.cat(parts, dim=-1)   # (H, W, emb_dim)

        if learnable_pe:
            # same trick as original Hyena -- z is an nn.Parameter with a very small lr
            # (1e-5) so it barely drifts from the theoretically motivated sin/cos init
            # allows small per-dataset corrections without throwing away the prior
            # to revert: just set learnable_pe=False, behavior is identical to before
            self.z = nn.Parameter(z)
            self.z._optim = {'lr': 1e-5}  # slow this parameter way down
        else:
            # default: completely fixed buffer, no gradient, base behavior unchanged
            self.register_buffer('z', z)

        # distance from (0,0) for exponential decay window
        # forces kernel to be compact -- same role as Hyena's ExponentialModulation
        dist = torch.sqrt(grid_i ** 2 + grid_j ** 2)   # (H, W)
        self.register_buffer('dist', dist)

        # learnable decay rate per output channel
        self.log_decay = nn.Parameter(torch.zeros(d_model))

        # FFN with sine activations: maps positional encoding to kernel values
        # sine activations are key -- they let the FFN represent high-frequency
        # spatial patterns that ReLU/GELU would struggle with
        act = SineActivation(dim=filter_order, w=w)
        layers = [nn.Linear(emb_dim, filter_order), act]
        for _ in range(kernel_layers):
            layers += [nn.Linear(filter_order, filter_order), act]
        layers.append(nn.Linear(filter_order, d_model, bias=False))
        self.ffn = nn.Sequential(*layers)

    def forward(self, H, W):
        # resize positional encoding if the feature map size changed (e.g. rectangular input)
        if H != self.img_size or W != self.img_size:
            z = F.interpolate(
                self.z.permute(2, 0, 1).unsqueeze(0),    # (1, emb_dim, H0, W0)
                size=(H, W), mode='bilinear', align_corners=False
            ).squeeze(0).permute(1, 2, 0)                # (H, W, emb_dim)
            dist = F.interpolate(
                self.dist.unsqueeze(0).unsqueeze(0),      # (1, 1, H0, W0)
                size=(H, W), mode='bilinear', align_corners=False
            ).squeeze()                                   # (H, W)
        else:
            z, dist = self.z, self.dist

        # generate full-rank 2D kernel from positional encodings
        h = self.ffn(z)    # (H, W, d_model)

        # exponential decay: kernel values far from (0,0) are suppressed
        # each channel can learn its own decay rate via log_decay
        decay = torch.exp(-self.log_decay.exp() * dist.unsqueeze(-1))   # (H, W, d_model)
        h = h * decay

        return h.permute(2, 0, 1)   # (d_model, H, W)


class HyenaBlock(nn.Module):
    """
    Single Hyena-2D block: prenorm -> short_conv -> fftconv -> residual add.

    This is the reusable building block -- no fuse, no skip concat with original
    input. Those belong in the outer wrapper (HyenaNeckConcatPE) so multiple
    blocks can be stacked before the final fuse, same as how S4ND stacks
    SequenceResidualBlocks before the detection head.

    Input and output are both (B, C, H, W) -- channel dim unchanged.
    """
    def __init__(self, c1, img_size=20, filter_order=16, bands=1, kernel_layers=2,
                 bidirectional=True, w=1, use_pe=False, use_gate=False, bi_fusion='add',
                 use_d_skip=False, learnable_pe=False, use_channel_mix_ffn=False,
                 kernel_order=1, gate_order=2):
        super().__init__()
        self.bidirectional = bidirectional
        self.use_pe = use_pe
        self.use_gate = use_gate
        self.kernel_order = kernel_order  # FFT conv passes (ungated path)
        self.gate_order = gate_order      # Hyena-style: gate_order-1 gate→FFT pairs + 1 final gate

        assert bi_fusion in ('add', 'concat'), f"bi_fusion must be 'add' or 'concat', got '{bi_fusion}'"
        self.bi_fusion = bi_fusion
        self.use_d_skip = use_d_skip

        # D skip: learnable per-channel direct feedthrough, same as SSM's D matrix
        # zero-init so at the start of training it contributes nothing -- base behavior preserved
        # both S4ND and Hyena have this (S4ND calls it self.D, Hyena calls it bias)
        if use_d_skip:
            self.D = nn.Parameter(torch.zeros(c1))

        # channel mixing FFN: LayerNorm -> Conv1x1(C->4C) -> GELU -> Conv1x1(4C->C)
        # same as transformer FFN / S4ND output_linear / MambND feedforward_channels
        # mixes channels at each spatial position after the global conv -- orthogonal to
        # our spatial mixing (the full-rank kernel handles positions, this handles channels)
        # use_channel_mix_ffn=False (default): disabled, base neck behavior unchanged
        # use_channel_mix_ffn=True: needed for backbone use where no prior channel mixing exists
        self.use_channel_mix_ffn = use_channel_mix_ffn
        if use_channel_mix_ffn:
            self.channel_mix_norm = nn.LayerNorm(c1)
            self.channel_mix_ffn = nn.Sequential(
                nn.Conv2d(c1, 4 * c1, kernel_size=1),   # expand channels 4x
                nn.GELU(),
                nn.Conv2d(4 * c1, c1, kernel_size=1),   # contract back to c1
            )

        # when bi_fusion='concat' we need a 1x1 conv to project 2C -> C after combining
        # fwd and bwd outputs -- only created when bidirectional=True
        # fuse_norm stabilizes FFT output scale before the learned fusion to prevent collapse
        if bidirectional and bi_fusion == 'concat':
            self.fuse_bi = nn.Conv2d(2 * c1, c1, kernel_size=1)
            self.fuse_norm = nn.LayerNorm(2 * c1)

        # learnable 2D positional embedding -- zero-init so training starts identical to no-PE
        if use_pe:
            self.pos_embed = nn.Parameter(torch.zeros(1, c1, img_size, img_size))

        # prenorm: LayerNorm over channel dim per spatial position (same as S4ND's prenorm=True)
        self.norm = nn.LayerNorm(c1)

        if use_gate:
            # gated path: gate_order-1 gate→FFT pairs + 1 final gate (Hyena convention)
            # gate_order=2 (default): gate→FFT→final_gate       (1 pair)
            # gate_order=3:           gate→FFT→gate→FFT→final_gate (2 pairs)
            # gate_order < 2 means 0 FFT convs — degenerate, not useful
            assert gate_order >= 2, f"gate_order must be >= 2 when use_gate=True (got {gate_order})"
            n_gate_pairs = gate_order - 1
            n_splits = (2 + 2 * n_gate_pairs) if bidirectional else (2 + n_gate_pairs)
            self.in_proj = nn.Conv2d(c1, n_splits * c1, kernel_size=1)
            self.out_proj = nn.Conv2d(c1, c1, kernel_size=1)
            self.short_conv = nn.Conv2d(n_splits * c1, n_splits * c1,
                                        kernel_size=3, padding=1, groups=n_splits * c1)
        else:
            # no gating: short conv only processes c1 channels
            self.short_conv = nn.Conv2d(c1, c1, kernel_size=3, padding=1, groups=c1)

        # filters: gate_order-1 for gated path (one per gate→FFT pair), kernel_order for ungated
        # defaults (gate_order=2, kernel_order=1): 1 filter each — identical to previous behavior
        n_filters = (gate_order - 1) if use_gate else kernel_order
        self.hyena_filters_fwd = nn.ModuleList([
            Hyena2DFilter(
                d_model=c1, img_size=img_size, bands=bands,
                filter_order=filter_order, kernel_layers=kernel_layers,
                w=w, learnable_pe=learnable_pe,
            ) for _ in range(n_filters)
        ])
        if bidirectional:
            self.hyena_filters_bwd = nn.ModuleList([
                Hyena2DFilter(
                    d_model=c1, img_size=img_size, bands=bands,
                    filter_order=filter_order, kernel_layers=kernel_layers,
                    w=w, learnable_pe=learnable_pe,
                ) for _ in range(n_filters)
            ])


    def forward(self, x):
        B, C, H, W = x.shape

        # add learnable 2D PE -- skip concat below uses original x (no PE)
        if self.use_pe:
            pos = F.interpolate(self.pos_embed, size=(H, W), mode='bilinear', align_corners=False)
            x_in = x + pos
        else:
            x_in = x

        residual = x_in

        # prenorm: permute to (B, H, W, C) for LayerNorm then back
        y = self.norm(x_in.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.use_gate:
            y = self.in_proj(y)       # (B, n_splits*C, H, W)
            y = self.short_conv(y)    # (B, n_splits*C, H, W)

            D = self.D if self.use_d_skip else None
            n_gate_pairs = self.gate_order - 1
            if self.bidirectional:
                # split: x0 | (gate_order-1) fwd gates | (gate_order-1) bwd gates | v
                # gate_order=2 (default): (x0, x_fwd1, x_bwd1, v) -- identical to before
                chunks = y.chunk(2 + 2 * n_gate_pairs, dim=1)
                x0     = F.silu(chunks[0])
                x_fwds = [F.silu(chunks[1 + o]) for o in range(n_gate_pairs)]
                x_bwds = [F.silu(chunks[1 + n_gate_pairs + o]) for o in range(n_gate_pairs)]
                v      = chunks[-1]

                y_fwd = v
                for o in range(n_gate_pairs):
                    y_fwd = fftconv2d(y_fwd * x_fwds[o], self.hyena_filters_fwd[o](H, W), D)
                y_bwd = v
                for o in range(n_gate_pairs):
                    y_bwd = fftconv2d(y_bwd * x_bwds[o], self.hyena_filters_bwd[o](H, W), D)

                if self.bi_fusion == 'concat':
                    cat_feat = torch.cat([y_fwd, y_bwd], dim=1)
                    cat_feat = self.fuse_norm(cat_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                    y = self.fuse_bi(cat_feat)
                else:
                    y = y_fwd + y_bwd
            else:
                # split: x0 | (gate_order-1) gates | v
                # gate_order=2 (default): (x0, x1, v) -- identical to before
                chunks = y.chunk(2 + n_gate_pairs, dim=1)
                x0      = F.silu(chunks[0])
                x_gates = [F.silu(chunks[1 + o]) for o in range(n_gate_pairs)]
                v       = chunks[-1]

                y_fwd = v
                for o in range(n_gate_pairs):
                    y_fwd = fftconv2d(y_fwd * x_gates[o], self.hyena_filters_fwd[o](H, W), D)
                y = y_fwd

            # x0 is the final gate -- gives the operator its non-linear character
            y = y * x0
            # y = F.gelu(y)  # optional GELU after gating (commented out, marginal difference)
            y = self.out_proj(y)

        else:
            y = self.short_conv(y)
            D = self.D if self.use_d_skip else None

            # kernel_order FFT conv passes chained without gates -- kernel_order=1 identical to before
            y_fwd = y
            for o in range(self.kernel_order):
                y_fwd = fftconv2d(y_fwd, self.hyena_filters_fwd[o](H, W), D)

            if self.bidirectional:
                y_bwd = y
                for o in range(self.kernel_order):
                    y_bwd = fftconv2d(y_bwd, self.hyena_filters_bwd[o](H, W), D)
                if self.bi_fusion == 'concat':
                    cat_feat = torch.cat([y_fwd, y_bwd], dim=1)
                    cat_feat = self.fuse_norm(cat_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                    y = self.fuse_bi(cat_feat)
                else:
                    y = y_fwd + y_bwd
            else:
                y = y_fwd

            # y = F.gelu(y)  # optional GELU (commented out, marginal difference)

        # additive residual: x = x + block(norm(x)), same as ViT wrapper
        y = y + residual

        # channel mixing FFN -- applied after residual, same pattern as S4ND/MambND/Hyena
        # prenorm -> expand -> GELU -> contract -> residual add
        # if use_channel_mix_ffn=False (default) this is skipped entirely, base behavior preserved
        if self.use_channel_mix_ffn:
            y = y + self.channel_mix_ffn(
                self.channel_mix_norm(y.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
            )

        return y


class HyenaNeckConcatPE(nn.Module):
    """
    Hyena-2D neck module -- drop-in replacement for S4NDNeckConcatPE.

    Wraps n_blocks HyenaBlocks (default 1) followed by a single skip concat
    with the original FPN features and a 1x1 fuse conv. With n_blocks=1 the
    behavior is identical to the original single-block design.

    Stacking multiple blocks (n_blocks > 1) follows S4ND's SequenceResidualBlock
    pattern -- each block does prenorm -> conv -> residual, and the fuse only
    happens once at the very end after all blocks have processed the features.

    Args match S4NDNeckConcatPE style so it works as a YAML drop-in:
      c1, c2, img_size, filter_order, bands, kernel_layers, bidirectional, w,
      use_pe, use_gate, bi_fusion, use_d_skip, learnable_pe, n_blocks
    """
    def __init__(self, c1, c2, img_size=20, filter_order=16, bands=1, kernel_layers=2,
                 bidirectional=True, w=1, use_pe=False, use_gate=False, bi_fusion='add',
                 use_d_skip=False, learnable_pe=False, n_blocks=1, use_channel_mix_ffn=False,
                 kernel_order=1, gate_order=2):
        super().__init__()

        # stack n_blocks independent HyenaBlocks -- each has its own parameters
        # with n_blocks=1 (default) this is identical to the old single-block design
        self.blocks = nn.ModuleList([
            HyenaBlock(
                c1=c1, img_size=img_size, filter_order=filter_order, bands=bands,
                kernel_layers=kernel_layers, bidirectional=bidirectional, w=w,
                use_pe=use_pe, use_gate=use_gate, bi_fusion=bi_fusion,
                use_d_skip=use_d_skip, learnable_pe=learnable_pe,
                use_channel_mix_ffn=use_channel_mix_ffn,
                kernel_order=kernel_order, gate_order=gate_order,
            )
            for _ in range(n_blocks)
        ])

        # fuse happens once at the end -- gives detection head access to both
        # global-context-enriched features and the raw FPN features separately
        self.fuse = nn.Conv2d(c1 + c1, c2, kernel_size=1)

    def forward(self, x):
        # keep original FPN features for the skip concat at the end
        x_orig = x

        # pass through all blocks sequentially -- each block does its own
        # prenorm + conv + residual, output feeds into the next block
        y = x
        for block in self.blocks:
            y = block(y)

        # concat enriched features with original FPN features and fuse
        return self.fuse(torch.cat([y, x_orig], dim=1))
