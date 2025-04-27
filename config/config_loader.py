import os
import yaml
from pathlib import Path

from .config_schema import Config


def load_config(config_path: str) -> Config:
    """Загружает конфигурацию из YAML-файла.

    Args:
        config_path: Путь к YAML-файлу с конфигурацией.

    Returns:
        Валидированная конфигурация.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as file:
        config_dict = yaml.safe_load(file)

    return Config(**config_dict)


def save_config(config: Config, config_path: str) -> None:
    """Сохраняет конфигурацию в YAML-файл.

    Args:
        config: Конфигурация для сохранения.
        config_path: Путь для сохранения конфигурации.
    """
    config_dict = config.dict()

    # Создаем директорию, если она не существует
    Path(os.path.dirname(config_path)).mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w', encoding='utf-8') as file:
        yaml.dump(config_dict, file, default_flow_style=False)