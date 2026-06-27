optimizer = {
    "adam":    "torch.optim.Adam",
    "adamw":   "torch.optim.AdamW",
    "rmsprop": "torch.optim.RMSprop",
    "sgd":     "torch.optim.SGD",
    "lamb":    "models.s4.src.utils.optim.lamb.JITLamb",
}

scheduler = {
    "constant":        "transformers.get_constant_schedule",
    "plateau":         "torch.optim.lr_scheduler.ReduceLROnPlateau",
    "step":            "torch.optim.lr_scheduler.StepLR",
    "multistep":       "torch.optim.lr_scheduler.MultiStepLR",
    "cosine":          "torch.optim.lr_scheduler.CosineAnnealingLR",
    "constant_warmup": "transformers.get_constant_schedule_with_warmup",
    "linear_warmup":   "transformers.get_linear_schedule_with_warmup",
    "cosine_warmup":   "transformers.get_cosine_schedule_with_warmup",
    "timm_cosine":     "models.s4.src.utils.optim.schedulers.TimmCosineLRScheduler",
}

callbacks = {
    "timer":                 "models.s4.src.callbacks.timer.Timer",
    "params":                "models.s4.src.callbacks.params.ParamsLog",
    "learning_rate_monitor": "pytorch_lightning.callbacks.LearningRateMonitor",
    "model_checkpoint":      "pytorch_lightning.callbacks.ModelCheckpoint",
    "early_stopping":        "pytorch_lightning.callbacks.EarlyStopping",
    "swa":                   "pytorch_lightning.callbacks.StochasticWeightAveraging",
    "rich_model_summary":    "pytorch_lightning.callbacks.RichModelSummary",
    "rich_progress_bar":     "pytorch_lightning.callbacks.RichProgressBar",
    "progressive_resizing":  "models.s4.src.callbacks.progressive_resizing.ProgressiveResizing",
    # "profiler": "pytorch_lightning.profilers.PyTorchProfiler",
}

model = {
    # Backbones from this repo
    "model":                 "models.s4.src.models.sequence.backbones.model.SequenceModel",
    "unet":                  "models.s4.src.models.sequence.backbones.unet.SequenceUNet",
    "sashimi":               "models.s4.src.models.sequence.backbones.sashimi.Sashimi",
    "sashimi_standalone":    "models.sashimi.sashimi.Sashimi",
    # Baseline RNNs
    "lstm":                  "models.s4.src.models.baselines.lstm.TorchLSTM",
    "gru":                   "models.s4.src.models.baselines.gru.TorchGRU",
    "unicornn":              "models.s4.src.models.baselines.unicornn.UnICORNN",
    "odelstm":               "models.s4.src.models.baselines.odelstm.ODELSTM",
    "lipschitzrnn":          "models.s4.src.models.baselines.lipschitzrnn.RnnModels",
    "stackedrnn":            "models.s4.src.models.baselines.samplernn.StackedRNN",
    "stackedrnn_baseline":   "models.s4.src.models.baselines.samplernn.StackedRNNBaseline",
    "samplernn":             "models.s4.src.models.baselines.samplernn.SampleRNN",
    "dcgru":                 "models.s4.src.models.baselines.dcgru.DCRNNModel_classification",
    "dcgru_ss":              "models.s4.src.models.baselines.dcgru.DCRNNModel_nextTimePred",
    # Baseline CNNs
    "ckconv":                "models.s4.src.models.baselines.ckconv.ClassificationCKCNN",
    "wavegan":               "models.s4.src.models.baselines.wavegan.WaveGANDiscriminator", # DEPRECATED
    "denseinception":        "models.s4.src.models.baselines.dense_inception.DenseInception",
    "wavenet":               "models.s4.src.models.baselines.wavenet.WaveNetModel",
    "torch/resnet2d":        "models.s4.src.models.baselines.resnet.TorchVisionResnet",  # 2D ResNet
    # Nonaka 1D CNN baselines
    "nonaka/resnet18":       "models.s4.src.models.baselines.nonaka.resnet.resnet1d18",
    "nonaka/inception":      "models.s4.src.models.baselines.nonaka.inception.inception1d",
    "nonaka/xresnet50":      "models.s4.src.models.baselines.nonaka.xresnet.xresnet1d50",
    # ViT Variants (note: small variant is taken from Tri, differs from original)
    "vit":                   "models.baselines.vit.ViT",
    "vit_s_16":              "models.s4.src.models.baselines.vit_all.vit_small_patch16_224",
    "vit_b_16":              "models.s4.src.models.baselines.vit_all.vit_base_patch16_224",
    # Timm models
    "timm/convnext_base":    "models.s4.src.models.baselines.convnext_timm.convnext_base",
    "timm/convnext_small":   "models.s4.src.models.baselines.convnext_timm.convnext_small",
    "timm/convnext_tiny":    "models.s4.src.models.baselines.convnext_timm.convnext_tiny",
    "timm/convnext_micro":   "models.s4.src.models.baselines.convnext_timm.convnext_micro",
    "timm/resnet50":         "models.s4.src.models.baselines.resnet_timm.resnet50", # Can also register many other variants in resnet_timm
    "timm/convnext_tiny_3d": "models.s4.src.models.baselines.convnext_timm.convnext3d_tiny",
    # Segmentation models
    "convnext_unet_tiny":    "models.s4.src.models.segmentation.convnext_unet.convnext_tiny_unet",
}

layer = {
    "id":         "models.s4.src.models.sequence.base.SequenceIdentity",
    "lstm":       "models.s4.src.models.baselines.lstm.TorchLSTM",
    "standalone": "models.s4.s4.S4Block",
    "s4d":        "models.s4.s4d.S4D",
    "ffn":        "models.s4.src.models.sequence.modules.ffn.FFN",
    "sru":        "models.s4.src.models.sequence.rnns.sru.SRURNN",
    "rnn":        "models.s4.src.models.sequence.rnns.rnn.RNN",  # General RNN wrapper
    "conv1d":     "models.s4.src.models.sequence.convs.conv1d.Conv1d",
    "conv2d":     "models.s4.src.models.sequence.convs.conv2d.Conv2d",
    "mha":        "models.s4.src.models.sequence.attention.mha.MultiheadAttention",
    "vit":        "models.s4.src.models.sequence.attention.mha.VitAttention",
    "performer":  "models.s4.src.models.sequence.attention.linear.Performer",
    "lssl":       "models.s4.src.models.sequence.modules.lssl.LSSL",
    "s4":         "models.s4.src.models.sequence.modules.s4block.S4Block",
    "fftconv":    "models.s4.src.models.sequence.kernels.fftconv.FFTConv",
    "s4nd":       "models.s4.src.models.sequence.modules.s4nd.S4ND",
    "mega":       "models.s4.src.models.sequence.modules.mega.MegaBlock",
    "h3":         "models.s4.src.models.sequence.experimental.h3.H3",
    "h4":         "models.s4.src.models.sequence.experimental.h4.H4",
    # 'packedrnn': 'models.sequence.rnns.packedrnn.PackedRNN',
}

layer_decay = {
    'convnext_timm_tiny': 'models.s4.src.models.baselines.convnext_timm.get_num_layer_for_convnext_tiny',
}

model_state_hook = {
    'convnext_timm_tiny_2d_to_3d': 'models.s4.src.models.baselines.convnext_timm.convnext_timm_tiny_2d_to_3d',
    'convnext_timm_tiny_s4nd_2d_to_3d': 'models.s4.src.models.baselines.convnext_timm.convnext_timm_tiny_s4nd_2d_to_3d',
}
