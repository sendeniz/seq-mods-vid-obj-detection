import hydra
from omegaconf import OmegaConf
from train import SequenceLightningModule


@hydra.main(config_path="configs", config_name="config")
def main(config):
    OmegaConf.set_struct(config, False)  # allow modifications
    model = SequenceLightningModule(config)
    print("Model initialized:", type(model))

if __name__ == "__main__":
    main()
