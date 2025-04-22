"""
Модуль для бинаризации целевой переменной.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Union, Optional
import logging

logger = logging.getLogger('tweet_training_service.data_processing.target_binarization')


class TargetBinarizer:
    """
    Класс для бинаризации целевой переменной на основе порогового значения.
    """

    def __init__(self, threshold: Union[int, float, str] = 'median'):
        """
        Инициализирует бинаризатор с указанным порогом.

        Args:
            threshold (Union[int, float, str]):
                Пороговое значение для бинаризации:
                - числовое значение: используется как порог напрямую
                - 'median': использовать медиану в качестве порога
                - 'mean': использовать среднее значение в качестве порога
                - 'percentile_XX': использовать XX-й перцентиль (например, 'percentile_75')
        """
        self.threshold = threshold
        self.calculated_threshold = None

    def fit(self, y: pd.Series) -> 'TargetBinarizer':
        """
        Вычисляет пороговое значение на основе данных.

        Args:
            y (pd.Series): Серия с целевой переменной.

        Returns:
            TargetBinarizer: Экземпляр класса для цепочки вызовов.

        Raises:
            ValueError: Если указан неподдерживаемый метод вычисления порога.
        """
        if isinstance(self.threshold, (int, float)):
            self.calculated_threshold = float(self.threshold)
        elif isinstance(self.threshold, str):
            if self.threshold == 'median':
                self.calculated_threshold = float(y.median())
            elif self.threshold == 'mean':
                self.calculated_threshold = float(y.mean())
            elif self.threshold.startswith('percentile_'):
                try:
                    percentile = int(self.threshold.split('_')[1])
                    if not 0 <= percentile <= 100:
                        raise ValueError(f"Перцентиль должен быть между 0 и 100, получено: {percentile}")
                    self.calculated_threshold = float(np.percentile(y, percentile))
                except (IndexError, ValueError) as e:
                    raise ValueError(f"Некорректный формат перцентиля: {self.threshold}. {str(e)}")
            else:
                raise ValueError(f"Неподдерживаемый метод вычисления порога: {self.threshold}")
        else:
            raise TypeError(f"Порог должен быть числом или строкой, получено: {type(self.threshold)}")

        logger.info(f"Вычислен порог для бинаризации: {self.calculated_threshold}")
        return self

    def transform(self, y: pd.Series) -> pd.Series:
        """
        Преобразует целевую переменную в бинарный формат.

        Args:
            y (pd.Series): Серия с целевой переменной.

        Returns:
            pd.Series: Бинаризованная целевая переменная (1 если значение >= порога, иначе 0).

        Raises:
            ValueError: Если порог не был вычислен (метод fit не был вызван).
        """
        if self.calculated_threshold is None:
            raise ValueError("Необходимо вызвать метод fit перед transform.")

        binary_y = (y >= self.calculated_threshold).astype(int)

        # Логируем распределение классов
        class_distribution = binary_y.value_counts(normalize=True) * 100
        logger.info(f"Распределение классов после бинаризации: "
                    f"0 (низкий потенциал): {class_distribution.get(0, 0):.2f}%, "
                    f"1 (высокий потенциал): {class_distribution.get(1, 0):.2f}%")

        return binary_y

    def fit_transform(self, y: pd.Series) -> pd.Series:
        """
        Вычисляет пороговое значение и преобразует целевую переменную.

        Args:
            y (pd.Series): Серия с целевой переменной.

        Returns:
            pd.Series: Бинаризованная целевая переменная.
        """
        return self.fit(y).transform(y)

    def get_threshold(self) -> Optional[float]:
        """
        Возвращает вычисленное пороговое значение.

        Returns:
            Optional[float]: Вычисленное пороговое значение или None, если fit не был вызван.
        """
        return self.calculated_threshold