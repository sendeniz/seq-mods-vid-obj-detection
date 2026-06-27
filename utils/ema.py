import torch
import torch.nn as nn
import math
from copy import deepcopy

def de_parallel(model):
    """De-parallelize a model"""
    return model.module if hasattr(model, 'module') else model

class EMA:
    """Memory-efficient in-place EMA that doesn't duplicate the model"""
    def __init__(self, model, decay=0.9999, tau=2000, updates=0, device=None):
        """Initialize EMA with parameter-level shadow values"""
        self.model = model
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp
        self.device = device
        
        # Store shadow parameters instead of full model copy
        self.shadow = {}
        self.backup = {}
        self._register_shadow_params()
        
    def _register_shadow_params(self):
        """Initialize shadow parameters from current model"""
        model = de_parallel(self.model)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and param.dtype.is_floating_point:
                    self.shadow[name] = param.data.clone()
        
    def update(self, model=None):
        """Update EMA parameters in-place"""
        if model is None:
            model = self.model
            
        self.updates += 1
        d = self.decay(self.updates)

        model = de_parallel(model)
        msd = model.state_dict()  # model state_dict
        
        with torch.no_grad():
            for k, v in msd.items():
                if k in self.shadow and v.dtype.is_floating_point:
                    self.shadow[k] = self.shadow[k] * d + v.detach() * (1 - d)

    def apply_shadow(self):
        """Swap model parameters with EMA shadow parameters for evaluation"""
        model = de_parallel(self.model)
        self.backup = {}
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow and param.requires_grad:
                    self.backup[name] = param.data.clone()
                    param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original model parameters after evaluation"""
        model = de_parallel(self.model)
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.backup and param.requires_grad:
                    param.data.copy_(self.backup[name])
            self.backup = {}

    def state_dict(self):
        return {
            'updates': self.updates,
            'shadow_params': self.shadow,
        }

    def load_state_dict(self, state_dict):
        self.updates = state_dict['updates']
        self.shadow = state_dict['shadow_params']