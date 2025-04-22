"""
Модуль для настройки логирования.
"""
import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional


def setup_logger(
        name: str,
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        max_file_size: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5
) -> logging.Logger:
    """
    Настраивает и возвращает логгер с заданными параметрами.

    Args:
        name (str): Имя логгера.
        log_level (str): Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file (str, optional): Путь к файлу логов. Если None, логи выводятся только в консоль.
        max_file_size (int): Максимальный размер файла логов в байтах.
        backup_count (int): Количество файлов для ротации.

    Returns:
        logging.Logger: Настроенный логгер.
    """
    # Преобразуем строковое представление уровня логирования в константу
    level = getattr(logging, log_level.upper())

    # Создаем логгер
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Очищаем старые обработчики, если они есть
    if logger.handlers:
        logger.handlers.clear()

    # Создаем форматтер
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Добавляем обработчик для вывода в консоль
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Если указан файл логов, добавляем обработчик для записи в файл с ротацией
    if log_file:
        # Создаем директорию для логов, если она не существует
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger