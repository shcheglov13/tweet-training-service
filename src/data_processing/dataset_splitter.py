"""
Модуль для разделения датасета на обучающую и тестовую выборки.
"""
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Union
from sklearn.model_selection import train_test_split, StratifiedKFold
import logging

logger = logging.getLogger('tweet_training_service.data_processing.dataset_splitter')


class DatasetSplitter:
    """
    Класс для разделения датасета на обучающую и тестовую выборки.
    """

    def __init__(
            self,
            test_size: float = 0.2,
            random_state: Optional[int] = 42,
            stratify: bool = True
    ):
        """
        Инициализирует сплиттер датасета.

        Args:
            test_size (float): Доля тестовой выборки (от 0 до 1).
            random_state (int, optional): Seed для генератора случайных чисел.
            stratify (bool): Использовать ли стратификацию для сохранения распределения классов.
        """
        if not 0 < test_size < 1:
            raise ValueError("Доля тестовой выборки должна быть в диапазоне (0, 1)")

        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify

    def split(
            self,
            x: pd.DataFrame,
            y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Разделяет датасет на обучающую и тестовую выборки.

        Args:
            x (pd.DataFrame): DataFrame с признаками.
            y (pd.Series): Серия с целевой переменной.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
                Кортеж (X_train, X_test, y_train, y_test).

        Raises:
            ValueError: Если X и y имеют разное количество строк.
        """
        if len(x) != len(y):
            raise ValueError(f"X и y должны иметь одинаковое количество строк. "
                             f"Получено: X - {len(x)}, y - {len(y)}")

        logger.info(f"Разделение датасета размером {len(x)} на обучающую и тестовую выборки "
                    f"(test_size={self.test_size}, stratify={self.stratify})")

        # Выполняем разделение
        if self.stratify:
            x_train, x_test, y_train, y_test = train_test_split(
                x, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y
            )
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                x, y,
                test_size=self.test_size,
                random_state=self.random_state
            )

        # Логируем результаты
        logger.info(f"Размер обучающей выборки: {len(x_train)}")
        logger.info(f"Размер тестовой выборки: {len(x_test)}")

        if self.stratify:
            train_class_distribution = y_train.value_counts(normalize=True) * 100
            test_class_distribution = y_test.value_counts(normalize=True) * 100

            logger.info(f"Распределение классов в обучающей выборке: "
                        f"0: {train_class_distribution.get(0, 0):.2f}%, "
                        f"1: {train_class_distribution.get(1, 0):.2f}%")

            logger.info(f"Распределение классов в тестовой выборке: "
                        f"0: {test_class_distribution.get(0, 0):.2f}%, "
                        f"1: {test_class_distribution.get(1, 0):.2f}%")

        return x_train, x_test, y_train, y_test
