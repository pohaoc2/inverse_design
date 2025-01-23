# config file for the BDM model
from dataclasses import dataclass
from typing import Any, Dict
import hydra
from omegaconf import DictConfig, OmegaConf
import copy


@dataclass
class LatticeConfig:
    # These must now be specified in config.yaml
    size: int
    initial_density: float

    def copy(self):
        return copy.deepcopy(self)


@dataclass
class RatesConfig:
    # These must now be specified in config.yaml
    proliferate: float
    death: float
    migrate: float

    def copy(self):
        return copy.deepcopy(self)


@dataclass
class OutputConfig:
    frequency: float
    max_time: float

    def copy(self):
        return copy.deepcopy(self)


@dataclass
class BDMConfig:
    lattice: LatticeConfig
    rates: RatesConfig
    output: OutputConfig
    verbose: bool

    def copy(self):
        return BDMConfig(
            lattice=self.lattice.copy(),
            rates=self.rates.copy(),
            output=self.output.copy(),
            verbose=self.verbose
        )

    @classmethod
    def from_dictconfig(cls, cfg: DictConfig) -> 'BDMConfig':
        return cls(
            lattice=LatticeConfig(**cfg.lattice),
            rates=RatesConfig(**cfg.rates),
            output=OutputConfig(**cfg.output),
            verbose=cfg.verbose
        )


@dataclass
class ParameterRange:
    min: float
    max: float


@dataclass
class ABCConfig:
    sobol_power: int
    epsilon: float
    parameter_ranges: Dict[str, ParameterRange]
    output_frequency: int
    @classmethod
    def from_dictconfig(cls, cfg: DictConfig) -> 'ABCConfig':
        parameter_ranges = {
            name: ParameterRange(**ranges)
            for name, ranges in cfg.parameter_ranges.items()
        }
        return cls(
            epsilon=cfg.epsilon,
            sobol_power=cfg.sobol_power,
            parameter_ranges=parameter_ranges,
            output_frequency=cfg.output_frequency
        )


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="bdm_config", node=BDMConfig)


@hydra.main(version_base=None, config_name="bdm_config")
def get_config(cfg: DictConfig) -> Any:
    return OmegaConf.to_container(cfg, resolve=True)


if __name__ == "__main__":
    get_config()
