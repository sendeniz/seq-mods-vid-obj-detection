"""Decoders that interface between targets and model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce

#import src.models.nn.utils as U
#import src.utils as utils

import models.s4.src.models.nn.utils as U
import models.s4.src.utils as utils

class Decoder(nn.Module):
    """Abstract class defining the interface for Decoders.

    TODO: is there a way to enforce the signature of the forward method?
    """

    def forward(self, x, **kwargs):
        """
        x: (batch, length, dim) input tensor
        state: additional state from the model backbone
        *args, **kwargs: additional info from the dataset

        Returns:
        y: output tensor
        *args: other arguments to pass into the loss function
        """
        return x

    def step(self, x):
        """
        x: (batch, dim)
        """
        return self.forward(x.unsqueeze(1)).squeeze(1)


class SequenceDecoder(Decoder):
    def __init__(
        self, d_model, d_output=None, l_output=None, use_lengths=False, mode="last"
    ):
        super().__init__()

        self.output_transform = nn.Identity() if d_output is None else nn.Linear(d_model, d_output)

        if l_output is None:
            self.l_output = None
            self.squeeze = False
        elif l_output == 0:
            # Equivalent to getting an output of length 1 and then squeezing
            self.l_output = 1
            self.squeeze = True
        else:
            assert l_output > 0
            self.l_output = l_output
            self.squeeze = False

        self.use_lengths = use_lengths
        self.mode = mode

        if mode == 'ragged':
            assert not use_lengths

    def forward(self, x, state=None, lengths=None, l_output=None):
        """
        x: (n_batch, l_seq, d_model)
        Returns: (n_batch, l_output, d_output)
        """

        if self.l_output is None:
            if l_output is not None:
                assert isinstance(l_output, int)  # Override by pass in
            else:
                # Grab entire output
                l_output = x.size(-2)
            squeeze = False
        else:
            l_output = self.l_output
            squeeze = self.squeeze

        if self.mode == "last":
            restrict = lambda x: x[..., -l_output:, :]
        elif self.mode == "first":
            restrict = lambda x: x[..., :l_output, :]
        elif self.mode == "pool":
            restrict = lambda x: (
                torch.cumsum(x, dim=-2)
                / torch.arange(
                    1, 1 + x.size(-2), device=x.device, dtype=x.dtype
                ).unsqueeze(-1)
            )[..., -l_output:, :]

            def restrict(x):
                L = x.size(-2)
                s = x.sum(dim=-2, keepdim=True)
                if l_output > 1:
                    c = torch.cumsum(x[..., -(l_output - 1) :, :].flip(-2), dim=-2)
                    c = F.pad(c, (0, 0, 1, 0))
                    s = s - c  # (B, l_output, D)
                    s = s.flip(-2)
                denom = torch.arange(
                    L - l_output + 1, L + 1, dtype=x.dtype, device=x.device
                )
                s = s / denom
                return s

        elif self.mode == "sum":
            restrict = lambda x: torch.cumsum(x, dim=-2)[..., -l_output:, :]
            # TODO use same restrict function as pool case
        elif self.mode == 'ragged':
            assert lengths is not None, "lengths must be provided for ragged mode"
            # remove any additional padding (beyond max length of any sequence in the batch)
            restrict = lambda x: x[..., : max(lengths), :]
        else:
            raise NotImplementedError(
                "Mode must be ['last' | 'first' | 'pool' | 'sum']"
            )

        # Restrict to actual length of sequence
        if self.use_lengths:
            assert lengths is not None
            x = torch.stack(
                [
                    restrict(out[..., :length, :])
                    for out, length in zip(torch.unbind(x, dim=0), lengths)
                ],
                dim=0,
            )
        else:
            x = restrict(x)

        if squeeze:
            assert x.size(-2) == 1
            x = x.squeeze(-2)

        x = self.output_transform(x)

        return x

    def step(self, x, state=None):
        # Ignore all length logic
        return self.output_transform(x)

class NDDecoder(Decoder):
    """Decoder for single target (e.g. classification or regression)."""
    def __init__(
        self, d_model, d_output=None, mode="pool"
    ):
        super().__init__()

        assert mode in ["pool", "full"]
        self.output_transform = nn.Identity() if d_output is None else nn.Linear(d_model, d_output)

        self.mode = mode

    def forward(self, x, state=None):
        """
        x: (n_batch, l_seq, d_model)
        Returns: (n_batch, l_output, d_output)
        """
        
        # single yolo final feture i.e., x coords 32, 19, 19, 256
        if self.mode == 'pool':
            x = reduce(x, 'b ... h -> b h', 'mean') 
            # pool: torch.Size([32, 256])
        
        # output transform batchsize, nclasses i.e., torch.Size([32, 1083])
        x = self.output_transform(x)
        return x

class Yolo1DDecoder(nn.Module):
    """
    Used for a single final raw feature vec of shape Batch, 19, 19 3, 
    after final predictions and reshape is done. 
    S4ND linear encoder turns it into B, 19, 19, D_model for example.
    Decoder that maps S4ND features (B, H, W, D_model)
    back to YOLO-style predictions (B, A, H, W, 1) using a 1x1 conv.
    """
    def __init__(self, d_model=256, n_anchors=3, out_dim=1):
        super().__init__()
        self.n_anchors = n_anchors
        self.out_dim = out_dim
        self.conv = nn.Conv2d(
            in_channels=d_model,
            out_channels=n_anchors * out_dim,
            kernel_size=1
        )

    def forward(self, x):
        """
        Args:
            x: (B=batchsize, Scale, Scale, D_model)
        Returns:
            (B=batchsize, A, Scale, Scale, out_dim)
        """
        B, H, W, D = x.shape

        # Convert to NCHW for conv
        x = x.permute(0, 3, 1, 2)           # (B, D_model, Scale, Scale)
        x = self.conv(x)                     # (B, A*out_dim, Scale, Scale)

        # Reshape to (B, A, H, W, out_dim)
        x = x.view(B, self.n_anchors, self.out_dim, H, W)
        x = x.permute(0, 1, 3, 4, 2)        # (B, A, Scale, Scale, out_dim)
        #print("out shape:", x.shape)
        return x
    
class YoloNDDecoder(nn.Module):
    """Decoupled decoder for S4ND features - separate for box or class"""
    def __init__(self, d_model: int = 512, output_channels: int = 64, pool: str = "None"):
        super().__init__()
        self.pool = pool
        in_channels = d_model + 1 if pool == "poolhwconcat" else d_model
        
        # Single output head (either box or class)
        self.conv_out = nn.Conv2d(in_channels, output_channels, kernel_size=1)
        
        # Gating mechanism (for SE-style or gated residual)
        if pool in ["gated", "gated_residual"]:
            # Squeeze-and-Excitation style gating
            reduction = 1
            self.gate_fc = nn.Sequential(
                nn.Linear(d_model, d_model // reduction),  # Squeeze
                nn.ReLU(inplace=True),
                nn.Linear(d_model // reduction, d_model),  # Excitation
                nn.Sigmoid()  # Gate values in [0, 1]
            )
        
        # Full spatial gating (no pooling)
        elif pool in ["gated_full", "gated_full_residual"]:
            # Learn spatial gate from both x and y
            self.spatial_gate = nn.Sequential(
                nn.Conv2d(d_model * 2, d_model, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(d_model, d_model, kernel_size=1),
                nn.Sigmoid()  # [B, C, H, W] gate values
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: latent from s4nd [B, H, W, D]
            y: original feature before latent [B, C, H, W]
        Returns:
            predictions [B, output_channels, H, W]
        """
        B, H, W, D = x.shape
        x = x.permute(0, 3, 1, 2)  # [B, D, H, W]
        
        if self.pool == 'None':
            x = y + x
            
        elif self.pool == 'pool':
            x = reduce(x, 'b c h w -> b c', 'mean')
            x = y + x[:, :, None, None]
            
        elif self.pool == 'poolhw':
            x = reduce(x, 'b c h w -> b h w', 'mean')
            x = y + x[:, None, :, :]
            
        elif self.pool == 'poolhwconcat':
            x = reduce(x, 'b c h w -> b h w', 'mean')
            x = torch.cat([y, x[:, None, :, :]], dim=1)  # B, C+1, H, W
            
        elif self.pool == 'gated':
            x = reduce(x, 'b c h w -> b c', 'mean')
            gate = self.gate_fc(x)
            gate = gate[:, :, None, None]
            x = y * gate  # SE-style
            
        elif self.pool == 'gated_residual':
            x = reduce(x, 'b c h w -> b c', 'mean')
            gate = self.gate_fc(x)
            gate = gate[:, :, None, None]
            x = y + x[:, :, None, None] * gate
            
        elif self.pool == 'gated_full':
            # Soft selection: choose between x and y per location
            combined = torch.cat([x, y], dim=1)  # [B, 2*C, H, W]
            gate = self.spatial_gate(combined)    # [B, C, H, W]
            # gate=1 → use y, gate=0 → use x
            x = y * gate + x * (1 - gate)
            
        elif self.pool == 'gated_full_residual':
            # Gated residual: always use y, add gated amount of x
            combined = torch.cat([x, y], dim=1)  # [B, 2*C, H, W]
            gate = self.spatial_gate(combined)    # [B, C, H, W]
            # Always keep y, gate controls how much x to add
            x = y + x * gate
        
        # Single prediction head
        x = x.to(self.conv_out.weight.dtype)
        out = self.conv_out(x)  # [B, output_channels, H, W]
        
        return out

class StateDecoder(Decoder):
    """Use the output state to decode (useful for stateful models such as RNNs or perhaps Transformer-XL if it gets implemented."""

    def __init__(self, d_model, state_to_tensor, d_output):
        super().__init__()
        self.output_transform = nn.Linear(d_model, d_output)
        self.state_transform = state_to_tensor

    def forward(self, x, state=None):
        return self.output_transform(self.state_transform(state))


class RetrievalHead(nn.Module):
    def __init__(self, d_input, d_model, n_classes, nli=True, activation="relu"):
        super().__init__()
        self.nli = nli

        if activation == "relu":
            activation_fn = nn.ReLU()
        elif activation == "gelu":
            activation_fn = nn.GELU()
        else:
            raise NotImplementedError

        if (
            self.nli
        ):  # Architecture from https://github.com/mlpen/Nystromformer/blob/6539b895fa5f798ea0509d19f336d4be787b5708/reorganized_code/LRA/model_wrapper.py#L74
            self.classifier = nn.Sequential(
                nn.Linear(4 * d_input, d_model),
                activation_fn,
                nn.Linear(d_model, n_classes),
            )
        else:  # Head from https://github.com/google-research/long-range-arena/blob/ad0ff01a5b3492ade621553a1caae383b347e0c1/lra_benchmarks/models/layers/common_layers.py#L232
            self.classifier = nn.Sequential(
                nn.Linear(2 * d_input, d_model),
                activation_fn,
                nn.Linear(d_model, d_model // 2),
                activation_fn,
                nn.Linear(d_model // 2, n_classes),
            )

    def forward(self, x):
        """
        x: (2*batch, dim)
        """
        outs = rearrange(x, "(z b) d -> z b d", z=2)
        outs0, outs1 = outs[0], outs[1]  # (n_batch, d_input)
        if self.nli:
            features = torch.cat(
                [outs0, outs1, outs0 - outs1, outs0 * outs1], dim=-1
            )  # (batch, dim)
        else:
            features = torch.cat([outs0, outs1], dim=-1)  # (batch, dim)
        logits = self.classifier(features)
        return logits


class RetrievalDecoder(Decoder):
    """Combines the standard FeatureDecoder to extract a feature before passing through the RetrievalHead."""

    def __init__(
        self,
        d_input,
        n_classes,
        d_model=None,
        nli=True,
        activation="relu",
        *args,
        **kwargs
    ):
        super().__init__()
        if d_model is None:
            d_model = d_input
        self.feature = SequenceDecoder(
            d_input, d_output=None, l_output=0, *args, **kwargs
        )
        self.retrieval = RetrievalHead(
            d_input, d_model, n_classes, nli=nli, activation=activation
        )

    def forward(self, x, state=None, **kwargs):
        x = self.feature(x, state=state, **kwargs)
        x = self.retrieval(x)
        return x

class PackedDecoder(Decoder):
    def forward(self, x, state=None):
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        return x


# For every type of encoder/decoder, specify:
# - constructor class
# - list of attributes to grab from dataset
# - list of attributes to grab from model

registry = {
    "stop": Decoder,
    "id": nn.Identity,
    "linear": nn.Linear,
    "sequence": SequenceDecoder,
    "nd": NDDecoder,
    "retrieval": RetrievalDecoder,
    "state": StateDecoder,
    "pack": PackedDecoder,
}
model_attrs = {
    "linear": ["d_output"],
    "sequence": ["d_output"],
    "nd": ["d_output"],
    "retrieval": ["d_output"],
    "state": ["d_state", "state_to_tensor"],
    "forecast": ["d_output"],
}

dataset_attrs = {
    "linear": ["d_output"],
    "sequence": ["d_output", "l_output"],
    "nd": ["d_output"],
    "retrieval": ["d_output"],
    # TODO rename d_output to n_classes?
    "state": ["d_output"],
    "forecast": ["d_output", "l_output"],
}


def _instantiate(decoder, model=None, dataset=None):
    """Instantiate a single decoder"""
    if decoder is None:
        return None

    if isinstance(decoder, str):
        name = decoder
    else:
        name = decoder["_name_"]

    # Extract arguments from attribute names
    dataset_args = utils.config.extract_attrs_from_obj(
        dataset, *dataset_attrs.get(name, [])
    )
    model_args = utils.config.extract_attrs_from_obj(model, *model_attrs.get(name, []))
    # Instantiate decoder
    obj = utils.instantiate(registry, decoder, *model_args, *dataset_args)
    return obj


def instantiate(decoder, model=None, dataset=None):
    """Instantiate a full decoder config, e.g. handle list of configs
    Note that arguments are added in reverse order compared to encoder (model first, then dataset)
    """
    decoder = utils.to_list(decoder)
    return U.PassthroughSequential(
        *[_instantiate(d, model=model, dataset=dataset) for d in decoder]
    )
