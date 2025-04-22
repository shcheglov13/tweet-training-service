"""
Модуль для загрузки конфигурации из YAML-файла.
"""
import os
import yaml
from typing import Dict, Any, Optional


class ConfigLoader:
    """
    Класс для загрузки конфигурации из YAML-файла.
    """

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Загружает конфигурацию из YAML-файла.

        Args:
            config_path (str): Путь к файлу конфигурации.

        Returns:
            Dict[str, Any]: Словарь с конфигурационными параметрами.

        Raises:
            FileNotFoundError: Если файл конфигурации не найден.
            yaml.YAMLError: Если возникла ошибка при парсинге YAML.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Файл конфигурации не найден: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as file:
            try:
                config = yaml.safe_load(file)
                return config
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Ошибка при парсинге YAML-файла: {e}")

    @staticmethod
    def get_nested_value(config: Dict[str, Any], path: str, default: Optional[Any] = None) -> Any:
        """
        Получает значение из вложенного словаря по указанному пути.

        Args:
            config (Dict[str, Any]): Словарь с конфигурацией.
            path (str): Путь к значению в формате 'key1.key2.key3'.
            default (Any, optional): Значение по умолчанию, если ключ не найден.

        Returns:
            Any: Значение из словаря или default, если ключ не найден.
        """
        keys = path.split('.')
        result = config

        for key in keys:
            if isinstance(result, dict) and key in result:
                result = result[key]
            else:
                return default

        return result