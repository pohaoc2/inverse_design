from conf.config import BDMConfig, ARCADEConfig
from .bdm.bdm import BDM
from .arcade.arcade import ARCADE
from typing import Union


class ModelFactory:
    """Factory class for creating model instances"""

    @staticmethod
    def create_model(model_type: str, config: Union[BDMConfig, ARCADEConfig]):
        """Create and return appropriate model instance based on model type
        Args:
            model_type: Type of model to create ("BDM", "ARCADE", etc.)
            config: Configuration for the model
        Returns:
            Instance of the specified model
        Raises:
            ValueError: If model_type is not supported
        """
        if model_type == "BDM":
            return BDM(config)
        elif model_type == "ARCADE":
            return ARCADE(config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
