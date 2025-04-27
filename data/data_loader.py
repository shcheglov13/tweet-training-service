import json
import logging
import os
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class DataLoader:
    """Класс для загрузки данных из JSON-файлов."""

    def __init__(self, log_level: str = "INFO") -> None:
        """
        Инициализирует загрузчик данных.

        Args:
            log_level: Уровень логирования.
        """
        self.logger = logger

    def load_json(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Загружает данные из JSON-файла.

        Args:
            file_path: Путь к JSON-файлу.

        Returns:
            Список словарей с данными.

        Raises:
            FileNotFoundError: Если файл не найден.
            json.JSONDecodeError: Если файл содержит некорректный JSON.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл не найден: {file_path}")

        self.logger.info(f"Загрузка данных из файла: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            self.logger.info(f"Загружено {len(data)} записей")
            return data
        except json.JSONDecodeError as e:
            self.logger.error(f"Ошибка при разборе JSON: {e}")
            raise