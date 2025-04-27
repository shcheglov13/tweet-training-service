import logging
import sys
import os
from pathlib import Path
from typing import Optional


def setup_logger(
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        console_output: bool = True,
        log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> logging.Logger:
    """
    Настраивает логгер для сервиса.

    Args:
        log_level: Уровень логирования ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL').
        log_file: Путь к файлу для записи логов. Если None, то логи записываются только в консоль.
        console_output: Флаг, указывающий, нужно ли выводить логи в консоль.
        log_format: Формат сообщений логов.

    Returns:
        Настроенный логгер.
    """
    # Преобразуем уровень логирования
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Некорректный уровень логирования: {log_level}")

    # Настраиваем корневой логгер
    root_logger = logging.getLogger()  # Получаем корневой логгер
    root_logger.setLevel(numeric_level)

    # Удаляем существующие обработчики, если они есть
    if root_logger.handlers:
        root_logger.handlers.clear()

    # Создаем форматтер
    formatter = logging.Formatter(log_format)

    # Добавляем обработчик для вывода в консоль
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Добавляем обработчик для записи в файл, если указан путь
    if log_file:
        # Создаем директорию для лог-файла, если она не существует
        log_dir = os.path.dirname(log_file)
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Создаем логгер проекта (для сохранения совместимости)
    logger = logging.getLogger("tweet_training_service")

    return logger