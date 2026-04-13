# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
from typing import Type, TypeVar

from omegaconf import DictConfig, ListConfig, OmegaConf

logger = logging.getLogger()

T = TypeVar("T")

def set_struct_recursively(cfg, strict: bool = True):
    # Set struct mode for the current level
    OmegaConf.set_struct(cfg, strict)

    # Traverse through nested dictionaries and lists
    if isinstance(cfg, DictConfig):
        for key, value in cfg.items():
            if isinstance(value, (DictConfig, ListConfig)):
                set_struct_recursively(value, strict)
    elif isinstance(cfg, ListConfig):
        for item in cfg:
            if isinstance(item, (DictConfig, ListConfig)):
                set_struct_recursively(item, strict)


def flatten_dict(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def dataclass_from_dict(cls: Type[T], data: dict, strict: bool = True) -> T:
    """
    Converts a dictionary to a dataclass instance, recursively for nested structures.
    """
    base = OmegaConf.structured(cls())
    OmegaConf.set_struct(base, strict)
    override = OmegaConf.create(data)
    return OmegaConf.to_object(OmegaConf.merge(base, override))


def dataclass_to_dict(dataclass_instance: T) -> dict:
    """
    Converts a dataclass instance to a dictionary, recursively for nested structures.
    """
    if isinstance(dataclass_instance, dict):
        return dataclass_instance

    return OmegaConf.to_container(
        OmegaConf.structured(dataclass_instance), resolve=True
    )


def load_config_file(config_file, dataclass_cls: Type[T]) -> T:
    config = OmegaConf.to_container(OmegaConf.load(config_file), resolve=True)
    return dataclass_from_dict(dataclass_cls, config)


def dump_config(config, path, log_config=True):
    yaml_dump = OmegaConf.to_yaml(OmegaConf.structured(config))
    with open(path, "w") as f:
        if log_config:
            logger.info("Using the following config for this run:")
            logger.info(yaml_dump)
        f.write(yaml_dump)
