import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from einops import rearrange, reduce

from models.s4.src.models.sequence.backbones.block import SequenceResidualBlock
from models.s4.src.models.nn import Normalization
from models.s4.src.tasks.decoders import NDDecoder, Yolo1DDecoder, YoloNDDecoder

class S4ND(nn.Module):
    def __init__(self, d_model=256, d_state=64, nclasses=30, l_max=None, channels=1, n_layers=6, dropout=0.1):
        super().__init__()

        self.encoder = nn.Linear(3, d_model)
        self.drop = nn.Identity()

        self.layers = nn.ModuleList([
            SequenceResidualBlock(
                d_input=d_model,
                prenorm=True,
                layer={
                    "_name_": "s4nd",
                    "d_state": d_state,
                    "channels": 1,
                    "bidirectional": True,
                    "activation": "gelu",
                    "final_act": "glu",
                    "initializer": None,
                    "weight_norm": False,
                    "n_ssm": 1,
                    "dt_min": 0.1,
                    "dt_max": 1.0,
                    "l_max": l_max,
                    "dropout": dropout,
                    "tie_dropout": True,
                    "linear": False,
                    "transposed": False,
                },
                residual={"_name_": "residual"},
                norm={"_name_": "layer"},
                pool={"_name_": "avg"},
                dropout=dropout,
                tie_dropout=True,
                transposed=False,
            )
            for _ in range(n_layers)
        ])

        self.norm = Normalization(d_model, _name_="layer")
        #print('DECODER 2:', self.decoder)

        self.decoder = Yolo1DDecoder(d_model=d_model, n_anchors=3, out_dim=1)

    def forward(self, x, t, state=None):
        x = x.squeeze(-1)
        x = x.permute(0, 2, 3, 1)
        x = self.encoder(x)
        x = self.drop(x)

        for layer in self.layers:
            x, state = layer(x)
        
        x = self.norm(x)
        x = self.decoder(x)
        
        return x, state


class S4ND_v2(nn.Module):
    """Single S4ND branch for either box or class prediction"""
    def __init__(self, d_model=256, d_state=64, l_max=None, n_layers=6,
                 dropout=0.1, channels=3, output_channels=64, encoder_type="conv1x1",
                 decoder_pool='pool', bidirectional=True, refine=False,
                 kernel_mode='dplr', bandlimit=None, disc=None):
        super().__init__()
        self.encoder_type = encoder_type
        bidirectional = bool(bidirectional)

        # Kernel-specific kwargs forwarded into SSMKernel.__init__ via **kernel_args.
        # Only included when set, so DPLR path is byte-identical to before.
        kernel_extra = {}
        if bandlimit is not None:
            kernel_extra["bandlimit"] = bandlimit
        if disc is not None:
            kernel_extra["disc"] = disc

        # Create encoder based on encoder_type
        if encoder_type == "linear":
            self.encoder = nn.Linear(channels, d_model)
        elif encoder_type == "conv1x1":
            self.encoder = nn.Conv2d(channels, d_model, kernel_size=1, stride=1, padding=0)
        elif encoder_type == "conv3x3":
            self.encoder = nn.Conv2d(channels, d_model, kernel_size=3, stride=1, padding=1)
        else:
            raise ValueError(f"Unsupported encoder type: {encoder_type}")

        self.drop = nn.Identity()

        # S4ND layers (import your SequenceResidualBlock here)
        self.layers = nn.ModuleList([
            SequenceResidualBlock(
                d_input=d_model,
                prenorm=True,
                bidirectional=bidirectional,
                layer={
                    "_name_": "s4nd",
                    "d_state": d_state,
                    "channels": 1,
                    "trank": 1, # default 1
                    "bidirectional": bidirectional,
                    "refine": refine,
                    "mode": kernel_mode,
                    "activation": "gelu",
                    "final_act": "glu",
                    "initializer": None,
                    "weight_norm": False,
                    "n_ssm": 1,
                    "dt_min": 0.1,
                    "dt_max": 1.0,
                    "l_max": l_max,
                    "dropout": dropout,
                    "tie_dropout": True,
                    "linear": False,
                    "transposed": False,
                    **kernel_extra,
                },
                residual={"_name_": "residual"},
                norm={"_name_": "layer"},
                pool={"_name_": "avg"},
                dropout=dropout,
                tie_dropout=True,
                transposed=False,
            )
            for _ in range(n_layers)
        ])
        
        self.norm = Normalization(d_model, _name_="layer")
        self.decoder = YoloNDDecoder(d_model=d_model, output_channels=output_channels, pool=decoder_pool)
    
    def forward(self, x, t=None, state=None):
        """
        Args:
            x: Input features [B, C, H, W] from co-adaptation
        Returns:
            predictions [B, output_channels, H, W], state
        """
        input_dtype = x.dtype
        
        # Encode features
        if self.encoder_type == "linear":
            #print('x shape non permuted:', x.shape)
            x = x.permute(0, 2, 3, 1)  # [B, H, W, C]
            #print('x shape permuted:', x.shape)
            x = self.encoder(x)  # [B, H, W, d_model]
            #print('x shape after encoder:', x.shape)
            encoded_features = x.permute(0, 3, 1, 2)  # [B, d_model, H, W]
            #print('x shape permuted 2:', x.shape)

        else:  # conv1x1 or conv3x3
            x = self.encoder(x)  # [B, d_model, H, W]
            encoded_features = x.clone()
            x = x.permute(0, 2, 3, 1)  # [B, H, W, d_model]
        
        # S4ND processing
        x = self.drop(x)
        #print('x shape after dropout before s4nd:', x.shape)
        for layer in self.layers:
            x, state = layer(x)
        
        # Normalize latent
        x = self.norm(x)
        x = x.to(input_dtype)
        
        # Decode with both latent and encoded features
        x = self.decoder(x, encoded_features)
        
        return x, state


class S4NDNeck(nn.Module):
    """
    S4ND neck module that replaces SPPELAN at P5 in the GELAN neck.

    Analogous to MambaNDNeck: provides global spatial context at the coarsest
    feature scale (P5, 20x20 at 640px input) before cross-scale FPN/PAN fusion.

    S4ND_v2 already fuses latent + encoded_features internally via YoloNDDecoder.
    An additional external skip (skip_proj(raw_input)) is added on top, mirroring
    the feature_fusion='add' residual in MambaNDNeck.

    Args:
        c1:       Input channels (256 for GELAN-S P5).
        c2:       Output channels (must equal c1 for drop-in SPPELAN replacement).
        img_size: Spatial size of the P5 feature map (20 for 640px input).
        d_model:  Internal S4ND embedding dimension.
        d_state:  SSM state dimension.
        n_layers: Number of S4ND layers.
    """

    def __init__(self, c1, c2, img_size=20, d_model=256, d_state=64, n_layers=1):
        super().__init__()
        # Named 's4nd_neck' so train.py _is_s4nd_param() catches 's4nd' in name → pg2
        self.s4nd_neck = S4ND_v2(
            d_model=d_model,
            d_state=d_state,
            l_max=[img_size, img_size],
            n_layers=n_layers,
            dropout=0.0,
            channels=c1,
            output_channels=c2,
            encoder_type='conv1x1',
            decoder_pool='pool',
        )
        # External skip: raw input → c2, added to S4ND output (mirrors MambaNDNeck)
        self.skip_proj = nn.Conv2d(c1, c2, kernel_size=1)

        # Small-but-nonzero init: SSM starts quiet but gradients still flow to upstream layers
        nn.init.normal_(self.s4nd_neck.decoder.conv_out.weight, mean=0.0, std=0.01)
        if self.s4nd_neck.decoder.conv_out.bias is not None:
            nn.init.zeros_(self.s4nd_neck.decoder.conv_out.bias)

    def forward(self, x):
        s4nd_out, _ = self.s4nd_neck(x, t=None, state=None)
        return s4nd_out + self.skip_proj(x)


class S4NDNeckConcat(nn.Module):
    """
    S4ND global context module placed after FPN/PAN, using concat fusion.

    Concatenates SSM global features with local FPN/PAN features, then
    projects back to the original channel count via 1x1 conv. This preserves
    both local (CNN) and global (SSM) representations separately, letting
    the detection head learn to selectively use each.

    Args:
        c1:       Input channels from FPN/PAN output.
        c2:       Output channels (equals c1 for drop-in use before DDetect).
        img_size: Spatial size of the feature map (80/40/20 for P3/P4/P5).
        d_model:  Internal S4ND embedding dimension.
        d_state:  SSM state dimension.
        n_layers: Number of S4ND layers.
    """

    def __init__(self, c1, c2, img_size=20, d_model=128, d_state=64, n_layers=1):
        super().__init__()
        self.s4nd_neck = S4ND_v2(
            d_model=d_model,
            d_state=d_state,
            l_max=[img_size, img_size],
            n_layers=n_layers,
            dropout=0.0,
            channels=c1,
            output_channels=c2,
            encoder_type='conv1x1',
            decoder_pool='None',
        )
        # Fuse concat(SSM output, local input) -> c2
        self.fuse = nn.Conv2d(c2 + c1, c2, kernel_size=1)

    def forward(self, x):
        s4nd_out, _ = self.s4nd_neck(x, t=None, state=None)
        return self.fuse(torch.cat([s4nd_out, x], dim=1))


class S4NDNeckConcatPE(nn.Module):
    """
    S4NDNeckConcat with learnable 2D positional embeddings added to the SSM input.

    Gives the SSM explicit spatial coordinate information so it can learn
    position-aware global dependencies ("object at top-left relates to object
    at bottom-right") rather than purely sequence-order dependencies.
    Skip connection receives clean FPN features (no positional embedding).

    Args: same as S4NDNeckConcat.
    """

    def __init__(self, c1, c2, img_size=20, d_model=128, d_state=64, n_layers=1, bidirectional=True):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, c1, img_size, img_size))
        self.s4nd_neck = S4ND_v2(
            d_model=d_model,
            d_state=d_state,
            l_max=[img_size, img_size],
            n_layers=n_layers,
            dropout=0.0,
            channels=c1,
            output_channels=c2,
            encoder_type='conv1x1',
            decoder_pool='None',
            bidirectional=bidirectional,
        )
        self.fuse = nn.Conv2d(c2 + c1, c2, kernel_size=1)

    def forward(self, x):
        _, _, H, W = x.shape
        pos = F.interpolate(self.pos_embed, size=(H, W), mode='bilinear', align_corners=False)
        s4nd_out, _ = self.s4nd_neck(x + pos, t=None, state=None)
        return self.fuse(torch.cat([s4nd_out, x], dim=1))


class S4NDNeckConcatPERefine(nn.Module):
    """
    S4NDNeckConcatPE with learnable 1D refine vectors (r_h, r_w) added to the
    S4ND 1D kernels before the outer product.

    Fixes the DPLR exponential decay constraint: K_h_eff = K_h + r_h and
    K_w_eff = K_w + r_w, giving (K_h+r_h) ⊗ (K_w+r_w) — all 4 terms.
    r_h and r_w are unconstrained learnable vectors, zero-initialized so
    training starts identical to S4NDNeckConcatPE.

    To revert: use S4NDNeckConcatPE or gelan-s-ssm-concat-s4nd-pe.yaml.
    """

    def __init__(self, c1, c2, img_size=20, d_model=128, d_state=64, n_layers=1, bidirectional=True):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, c1, img_size, img_size))
        self.s4nd_neck = S4ND_v2(
            d_model=d_model,
            d_state=d_state,
            l_max=[img_size, img_size],
            n_layers=n_layers,
            dropout=0.0,
            channels=c1,
            output_channels=c2,
            encoder_type='conv1x1',
            decoder_pool='None',
            bidirectional=bidirectional,
            refine=True,
        )
        self.fuse = nn.Conv2d(c2 + c1, c2, kernel_size=1)

    def forward(self, x):
        _, _, H, W = x.shape
        pos = F.interpolate(self.pos_embed, size=(H, W), mode='bilinear', align_corners=False)
        s4nd_out, _ = self.s4nd_neck(x + pos, t=None, state=None)
        return self.fuse(torch.cat([s4nd_out, x], dim=1))


class S4NDNeckConcatPEDSS(nn.Module):
    """
    S4NDNeckConcatPE with diagonal SSM kernel (DSS / S4D) instead of DPLR.

    Replaces the DPLR-HiPPO-Legendre basis (designed for causal 1D signals,
    forces exponential decay) with a diagonal-SSM basis. With bandlimit set,
    high-frequency modes are masked, leaving smooth bandlimited 2D kernels —
    the inductive bias the S4ND paper itself argues is correct for vision
    (Section 4.2).

    Addresses the outer-product suppression problem at its source: K_h and K_w
    are no longer constrained to decay exponentially, so K_h[i] x K_w[j] is
    non-zero across the full 2D extent.

    To revert: use S4NDNeckConcatPE (DPLR default).
    """

    def __init__(self, c1, c2, img_size=20, d_model=128, d_state=64, n_layers=1,
                 bidirectional=True, bandlimit=0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, c1, img_size, img_size))
        self.s4nd_neck = S4ND_v2(
            d_model=d_model,
            d_state=d_state,
            l_max=[img_size, img_size],
            n_layers=n_layers,
            dropout=0.0,
            channels=c1,
            output_channels=c2,
            encoder_type='conv1x1',
            decoder_pool='None',
            bidirectional=bidirectional,
            kernel_mode='diag',
            bandlimit=bandlimit,
        )
        self.fuse = nn.Conv2d(c2 + c1, c2, kernel_size=1)

    def forward(self, x):
        _, _, H, W = x.shape
        pos = F.interpolate(self.pos_embed, size=(H, W), mode='bilinear', align_corners=False)
        s4nd_out, _ = self.s4nd_neck(x + pos, t=None, state=None)
        return self.fuse(torch.cat([s4nd_out, x], dim=1))


class S4NDNeckConcatCS(nn.Module):
    """
    S4ND global context module with cross-scale semantic injection.

    Before SSM processing, injects P5 semantic context (global avg pooled and
    projected) into the current scale features. This gives the SSM access to
    coarse semantic information (what objects exist globally) before modeling
    long-range spatial dependencies at finer scales.

    P3 and P4 use this module (they benefit from P5 semantic context).
    P5 uses S4NDNeckConcat (it is already the semantic scale).

    Args:
        c1:           Input channels from FPN/PAN output (current scale).
        c2:           Output channels (equals c1).
        img_size:     Spatial size of the feature map (80 for P3, 40 for P4).
        d_model:      Internal S4ND embedding dimension.
        d_state:      SSM state dimension.
        n_layers:     Number of S4ND layers.
        ctx_channels: P5 feature channels (256 for GELAN-S), injected via parse_model.
    """

    def __init__(self, c1, c2, img_size=20, d_model=128, d_state=64, n_layers=1, ctx_channels=256):
        super().__init__()
        self.ctx_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ctx_channels, c1, kernel_size=1),
        )
        self.s4nd_neck = S4ND_v2(
            d_model=d_model,
            d_state=d_state,
            l_max=[img_size, img_size],
            n_layers=n_layers,
            dropout=0.0,
            channels=c1,
            output_channels=c2,
            encoder_type='conv1x1',
            decoder_pool='None',
        )
        self.fuse = nn.Conv2d(c2 + c1, c2, kernel_size=1)

    def forward(self, inputs):
        x, ctx = inputs[0], inputs[1]   # x: current scale FPN, ctx: P5 features
        semantic = self.ctx_proj(ctx)   # [B, c1, 1, 1] global semantic prior
        x_enriched = x + semantic       # broadcast to [B, c1, H, W]
        s4nd_out, _ = self.s4nd_neck(x_enriched, t=None, state=None)
        return self.fuse(torch.cat([s4nd_out, x], dim=1))


"""
if __name__ == "__main__":
    print("Testing S4ND_v2")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    lr = 1e-3
    batch_size = 2
    weight_decay = 0.05
    dropout = 0.1
    n_layers = 6
    d_model = 256 * 2
    nc = 80
    reg_max = 16
    l_max = [32, 32]
    channels = 128
    
    print(f"Model configuration:")
    print(f"  d_model: {d_model}")
    print(f"  n_layers: {n_layers}")
    print(f"  dropout: {dropout}")
    print(f"  nc: {nc}")
    print(f"  reg_max: {reg_max}")
    print(f"  l_max: {l_max}")
    print(f"  channels: {channels}")
    
    print("Creating model...")
    model = S4ND_v2(
        d_model=d_model, 
        n_layers=n_layers, 
        dropout=dropout,
        nc=nc,
        reg_max=reg_max,
        l_max=l_max,
        channels=channels
    ).to(device)
    
    print("Model created successfully!")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print(f"Creating fake input: [{batch_size}, {channels}, {l_max[0]}, {l_max[1]}]")
    x = torch.randn(batch_size, channels, l_max[0], l_max[1]).to(device)
    print(f"Input shape: {x.shape}")
    print(f"Input device: {x.device}")
    
    print("Running forward pass...")
    
    try:
        model.eval()
        with torch.no_grad():
            output, state = model(x, t=None, state=None)
        
        print("Forward pass successful!")
        
        expected_channels = reg_max * 4 + nc
        print(f"Output analysis:")
        print(f"  Shape: {output.shape}")
        print(f"  Expected: [{batch_size}, {expected_channels}, {l_max[0]}, {l_max[1]}]")
        print(f"  Device: {output.device}")
        print(f"  Dtype: {output.dtype}")
        print(f"  Min value: {output.min().item():.4f}")
        print(f"  Max value: {output.max().item():.4f}")
        print(f"  Mean value: {output.mean().item():.4f}")
        
        assert output.shape == (batch_size, expected_channels, l_max[0], l_max[1])
        print(f"  Shape correct!")
        
        assert output.device == x.device
        print(f"  Device correct!")
        
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        print(f"  No NaN/Inf values!")
        
        box_pred = output[:, :reg_max*4, :, :]
        cls_pred = output[:, reg_max*4:, :, :]
        
        print(f"Prediction split:")
        print(f"  Box predictions: {box_pred.shape} (for DFL)")
        print(f"  Class predictions: {cls_pred.shape} (for classes)")
        
        print("All tests passed!")
        print("The S4ND_v2 module is working correctly!")
        print("Ready to integrate into YOLO training pipeline.")
        
    except RuntimeError as e:
        print("Forward pass failed!")
        print(f"Error: {e}")
        
        if "device" in str(e).lower():
            print("Device mismatch detected!")
            print("This is the S4 SSM kernel device issue.")
            print("Fix: Add this at the start of S4ND_v2.forward():")
            print("    device = x.device")
            print("    if next(self.parameters()).device != device:")
            print("        self.to(device)")
        
        raise
    
    except Exception as e:
        print("Unexpected error!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        raise
"""