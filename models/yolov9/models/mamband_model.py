import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Add mamba-nd directory to sys.path so image_classification imports resolve
_MAMBAND_DIR = Path(__file__).parent / 'mamba-nd'
if str(_MAMBAND_DIR) not in sys.path:
    sys.path.insert(0, str(_MAMBAND_DIR))

from image_classification.src.mamba import Mamba2DModel


class MambaND_v2(nn.Module):
    """
    Wrapper around Mamba2DModel for use as a detection-head branch in MambaNDDDetect.

    Replaces S4ND_v2: takes a [B, channels, H, W] feature map (post co-adaptation),
    processes it through Mamba2D scanning blocks, and projects to task-specific
    output channels via a 1x1 conv.

    Args:
        d_model:         Internal Mamba embedding dimension.
        d_state:         SSM state dimension (Mamba default is 16).
        img_size:        Spatial size of the input feature map (H == W assumed).
                         Must match the actual feature map: 80 for P3, 40 for P4, 20 for P5.
        n_layers:        Number of (forward, backward) scanning block pairs.
                         Internally Mamba2DModel doubles this: n_layers*2 total blocks.
        dropout:         Drop rate applied to both drop_rate and drop_path_rate.
        channels:        Input channel count (c2 for box branch, c3 for cls branch).
        output_channels: Output channel count (reg_max*4 for box, nc for cls).
        feature_fusion:  How to fuse local CNN features with Mamba global output.
                         'none'   - Mamba output only (no skip).
                         'add'    - element-wise add skip_proj(input) + proj(mamba_out).
                         'concat' - concat [mamba_out, input] then project.
    """

    def __init__(self, d_model=128, d_state=16, img_size=20,
                 n_layers=1, dropout=0.0, channels=64,
                 output_channels=64, feature_fusion='none'):
        super().__init__()

        assert feature_fusion in ('none', 'add', 'concat'), \
            f"feature_fusion must be 'none', 'add', or 'concat', got '{feature_fusion}'"

        self.feature_fusion = feature_fusion

        # Mamba2DModel prints a full parameter table in __init__; silence it
        _orig_count = Mamba2DModel.count_parameters
        Mamba2DModel.count_parameters = lambda self, model=None: None

        self.mamba = Mamba2DModel(
            arch='small',
            img_size=img_size,    # positional embedding size == feature map H/W
            patch_size=1,         # one token per spatial location (no downsampling)
            in_channels=channels,
            embed_dims=d_model,
            num_layers=n_layers,  # Mamba2DModel doubles this internally
            out_type='featmap',   # returns [B, d_model, H, W]
            with_cls_token=False,
            final_norm=True,
            fused_add_norm=False,
            d_state=d_state,
            is_2d=False,          # use sequence-style (not 2D conv-style) blocks
            has_reverse=True,     # alternate forward/backward scanning
            has_transpose=True,   # alternate row/column scanning
            constant_dim=True,    # keep d_model constant across all layers
            downsample=(),        # no spatial downsampling inside the module
            drop_rate=dropout,
            drop_path_rate=dropout,
        )

        Mamba2DModel.count_parameters = _orig_count  # restore for other callers

        if feature_fusion == 'concat':
            # project concat([mamba_out, input]) -> output_channels
            self.proj = nn.Conv2d(d_model + channels, output_channels, kernel_size=1)
        else:
            # project mamba_out -> output_channels
            self.proj = nn.Conv2d(d_model, output_channels, kernel_size=1)

        if feature_fusion == 'add':
            # parallel projection of local CNN features -> output_channels
            self.skip_proj = nn.Conv2d(channels, output_channels, kernel_size=1)

        # Small-but-nonzero init: SSM starts quiet but gradients still flow to upstream layers
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.01)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x, t=None, state=None):
        """
        Args:
            x: [B, channels, H, W]  feature map from co-adaptation layers
        Returns:
            out:   [B, output_channels, H, W]
            state: None (kept for API compatibility with S4ND_v2)
        """
        local = x
        mamba_out = self.mamba(x)[0]  # [B, d_model, H, W]

        if self.feature_fusion == 'add':
            out = self.proj(mamba_out) + self.skip_proj(local)
        elif self.feature_fusion == 'concat':
            out = self.proj(torch.cat([mamba_out, local], dim=1))
        else:
            out = self.proj(mamba_out)

        return out, state


class MambaNDNeck(nn.Module):
    """
    MambaND neck module that replaces SPPELAN at P5 in the GELAN neck.

    Analogous to the AIFI transformer encoder in RT-DETR: provides global
    spatial context at the coarsest feature scale (P5, 20x20 at 640px input)
    before cross-scale fusion in the FPN/PAN layers.

    Args:
        c1:       Input channels (from previous layer, e.g. 256 for GELAN-S).
        c2:       Output channels (must equal c1 for drop-in SPPELAN replacement).
        img_size: Spatial size of the P5 feature map (20 for 640px input).
        d_model:  Internal Mamba embedding dimension.
        d_state:  SSM state dimension.
        n_layers: Number of (forward+backward) scanning block pairs.
    """

    def __init__(self, c1, c2, img_size=20, d_model=256, d_state=16, n_layers=1):
        super().__init__()
        # Named 'mamband_neck' so train_mamband.py _is_mamba_param() routes it to pg2
        self.mamband_neck = MambaND_v2(
            d_model=d_model,
            d_state=d_state,
            img_size=img_size,
            n_layers=n_layers,
            dropout=0.0,
            channels=c1,
            output_channels=c2,
            feature_fusion='add',
        )

    def forward(self, x):
        out, _ = self.mamband_neck(x)
        return out


class MambaNDNeckConcat(nn.Module):
    """
    MambaND global context module placed after FPN/PAN, using concat fusion.

    Concatenates SSM global features with local FPN/PAN features, then
    projects back to the original channel count via 1x1 conv. This preserves
    both local (CNN) and global (SSM) representations separately, letting
    the detection head learn to selectively use each.

    Args:
        c1:       Input channels from FPN/PAN output.
        c2:       Output channels (equals c1 for drop-in use before DDetect).
        img_size: Spatial size of the feature map (80/40/20 for P3/P4/P5).
        d_model:  Internal Mamba embedding dimension.
        d_state:  SSM state dimension.
        n_layers: Number of (forward+backward) scanning block pairs.
    """

    def __init__(self, c1, c2, img_size=20, d_model=128, d_state=32, n_layers=1):
        super().__init__()
        self.mamband_neck = MambaND_v2(
            d_model=d_model,
            d_state=d_state,
            img_size=img_size,
            n_layers=n_layers,
            dropout=0.0,
            channels=c1,
            output_channels=c2,
            feature_fusion='none',
        )
        # Fuse concat(SSM output, local input) -> c2
        self.fuse = nn.Conv2d(c2 + c1, c2, kernel_size=1)

    def forward(self, x):
        ssm_out, _ = self.mamband_neck(x)
        return self.fuse(torch.cat([ssm_out, x], dim=1))


class MambaNDNeckConcatPE(nn.Module):
    """MambaND neck with learnable 2D positional embeddings, identical to S4NDNeckConcatPE."""

    def __init__(self, c1, c2, img_size=20, d_model=128, d_state=32, n_layers=1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, c1, img_size, img_size))
        self.mamband_neck = MambaND_v2(
            d_model=d_model,
            d_state=d_state,
            img_size=img_size,
            n_layers=n_layers,
            dropout=0.0,
            channels=c1,
            output_channels=c2,
            feature_fusion='none',
        )
        self.fuse = nn.Conv2d(c2 + c1, c2, kernel_size=1)

    def forward(self, x):
        _, _, H, W = x.shape
        pos = F.interpolate(self.pos_embed, size=(H, W), mode='bilinear', align_corners=False)
        ssm_out, _ = self.mamband_neck(x + pos)
        return self.fuse(torch.cat([ssm_out, x], dim=1))
