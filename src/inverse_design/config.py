# config file for the BDM model
from dataclasses import dataclass
from typing import Any
import hydra
from omegaconf import DictConfig, OmegaConf

@dataclass
class LatticeConfig:
    # These must now be specified in config.yaml
    size: int
    initial_density: float

@dataclass
class RatesConfig:
    # These must now be specified in config.yaml
    proliferate: float
    death: float
    migrate: float

@dataclass
class BDMConfig:
    lattice: LatticeConfig
    rates: RatesConfig

cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="bdm_config", node=BDMConfig)

@hydra.main(version_base=None, config_name="bdm_config")
def get_config(cfg: DictConfig) -> Any:
    return OmegaConf.to_container(cfg, resolve=True)

if __name__ == "__main__":
    get_config() 