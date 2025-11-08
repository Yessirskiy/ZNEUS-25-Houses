import importlib
import inspect
from typing import Type, Dict
import torch.nn as nn


def load_models(module_path: str) -> Dict[str, Type[nn.Module]]:
    """
    Load nn.Module classes from a Python module and return a dict:
    {class.NAME : class}

    Args:
        module_path (Path): Python module path

    Returns:
        Dict[str, Type[nn.Module]]: Mapping of class name to class object
    """
    models_dict: Dict[str, Type[nn.Module]] = {}

    module = importlib.import_module(module_path)

    for name, obj in inspect.getmembers(module, inspect.isclass):
        if issubclass(obj, nn.Module) and obj.__module__ == module.__name__:
            model_name = getattr(obj, "NAME", None)
            if model_name is not None:
                models_dict[model_name] = obj

    return models_dict
