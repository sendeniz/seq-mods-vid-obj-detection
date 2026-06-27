import sys
import torch
import torch.nn as nn
from pathlib import Path

# Add mamba-nd directory to sys.path so image_classification imports resolve
_MAMBAND_DIR = Path(__file__).parent / 'yolov9/models/mamba-nd'
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
    """

    def __init__(self, d_model=128, d_state=16, img_size=20,
                 n_layers=1, dropout=0.0, channels=64,
                 output_channels=64):
        super().__init__()

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

        # Task-specific projection: d_model -> output_channels
        self.proj = nn.Conv2d(d_model, output_channels, kernel_size=1)

    def forward(self, x, t=None, state=None):
        """
        Args:
            x: [B, channels, H, W]  feature map from co-adaptation layers
        Returns:
            out:   [B, output_channels, H, W]
            state: None (kept for API compatibility with S4ND_v2)
        """
        x = self.mamba(x)[0]  # Mamba2DModel returns a tuple; [0] is the featmap
        x = self.proj(x)
        return x, state
