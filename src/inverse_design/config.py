# config file for the BDM model
from dataclasses import dataclass
from typing import Any, Dict
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
class OutputConfig:
    frequency: float


@dataclass
class BDMConfig:
    lattice: LatticeConfig
    rates: RatesConfig
    output: OutputConfig

    @classmethod
    def from_dictconfig(cls, cfg: DictConfig) -> 'BDMConfig':
        return cls(
            lattice=LatticeConfig(**cfg.lattice),
            rates=RatesConfig(**cfg.rates),
            output=OutputConfig(**cfg.output)
        )


@dataclass
class ParameterRange:
    min: float
    max: float


@dataclass
class ABCConfig:
    num_samples: int
    epsilon: float
    parameter_ranges: Dict[str, ParameterRange]

    @classmethod
    def from_dictconfig(cls, cfg: DictConfig) -> 'ABCConfig':
        parameter_ranges = {
            name: ParameterRange(**ranges)
            for name, ranges in cfg.parameter_ranges.items()
        }
        return cls(
            num_samples=cfg.num_samples,
            epsilon=cfg.epsilon,
            parameter_ranges=parameter_ranges
        )


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="bdm_config", node=BDMConfig)


@hydra.main(version_base=None, config_name="bdm_config")
def get_config(cfg: DictConfig) -> Any:
    return OmegaConf.to_container(cfg, resolve=True)


if __name__ == "__main__":
    get_config()
