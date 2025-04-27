"""Модуль для работы с конфигурацией сервиса."""

from .config_schema import Config
from .config_loader import load_config, save_config

__all__ = ["Config", "load_config", "save_config"]