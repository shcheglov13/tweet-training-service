"""
Модуль для отбора признаков на основе корреляционного анализа.
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Set, Optional
import logging

logger = logging.getLogger('tweet_training_service.data_processing.feature_selection')

class FeatureSelector:
    """
    Класс для отбора признаков на основе корреляционного анализа.
    """

    def __init__(self, correlation_threshold: float = 0.95, target_column: Optional[str] = None):
        """
        Инициализирует селектор признаков.

        Args:
            correlation_threshold (float): Пороговое значение корреляции для исключения признаков.
            target_column (str, optional): Название колонки с целевой переменной.
                Если указано, признаки с низкой корреляцией с целевой переменной
                будут иметь приоритет для сохранения.
        """
        if not 0 <= correlation_threshold <= 1:
            raise ValueError("Пороговое значение корреляции должно быть в диапазоне [0, 1]")

        self.correlation_threshold = correlation_threshold
        self.target_column = target_column
        self.selected_features = None
        self.removed_features = None

    def _find_correlated_features(self, correlation_matrix: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """
        Находит пары признаков с корреляцией выше порогового значения.

        Args:
            correlation_matrix (pd.DataFrame): Матрица корреляции.

        Returns:
            List[Tuple[str, str, float]]: Список кортежей (признак1, признак2, корреляция).
        """
        correlated_features = []

        # Перебираем верхний треугольник матрицы корреляции
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                feature1 = correlation_matrix.columns[i]
                feature2 = correlation_matrix.columns[j]

                # Игнорируем целевую переменную, если она указана
                if self.target_column and (feature1 == self.target_column or feature2 == self.target_column):
                    continue

                correlation = correlation_matrix.iloc[i, j]

                # Если корреляция по модулю больше порога, добавляем пару в список
                if abs(correlation) >= self.correlation_threshold:
                    correlated_features.append((feature1, feature2, correlation))

        # Сортируем по убыванию абсолютного значения корреляции
        correlated_features.sort(key=lambda x: abs(x[2]), reverse=True)

        return correlated_features

    def fit(self, X: pd.DataFrame) -> 'FeatureSelector':
        """
        Выполняет корреляционный анализ и определяет признаки для удаления.

        Args:
            X (pd.DataFrame): DataFrame с признаками.

        Returns:
            FeatureSelector: Экземпляр класса для цепочки вызовов.
        """
        logger.info(f"Начало корреляционного анализа для {X.shape[1]} признаков.")

        # Создаем копию датафрейма с числовыми признаками
        numeric_features = X.select_dtypes(include=[np.number])

        if numeric_features.shape[1] == 0:
            logger.warning("В датафрейме отсутствуют числовые признаки.")
            self.selected_features = list(X.columns)
            self.removed_features = []
            return self

        # Вычисляем матрицу корреляции
        correlation_matrix = numeric_features.corr()

        # Находим пары сильно коррелирующих признаков
        correlated_pairs = self._find_correlated_features(correlation_matrix)

        if not correlated_pairs:
            logger.info(f"Не найдено признаков с корреляцией выше {self.correlation_threshold}.")
            self.selected_features = list(X.columns)
            self.removed_features = []
            return self

        # Логируем найденные пары
        logger.info(f"Найдено {len(correlated_pairs)} пар признаков с корреляцией выше {self.correlation_threshold}.")

        # Для удаления отмечаем признаки, которые уже решили удалить
        features_to_remove: Set[str] = set()

        # Если указана целевая переменная, вычисляем корреляцию признаков с целевой
        target_correlations = {}
        if self.target_column and self.target_column in X.columns:
            for feature in numeric_features.columns:
                if feature != self.target_column:
                    target_correlations[feature] = abs(
                        correlation_matrix.loc[feature, self.target_column]
                        if self.target_column in correlation_matrix.index else 0
                    )

        # Обрабатываем каждую пару коррелирующих признаков
        for feature1, feature2, correlation in correlated_pairs:
            # Если один из признаков уже удален, пропускаем
            if feature1 in features_to_remove or feature2 in features_to_remove:
                continue

            # Решаем, какой признак удалить на основе корреляции с целевой переменной
            if self.target_column and self.target_column in X.columns:
                # Удаляем признак с меньшей корреляцией с целевой
                if target_correlations.get(feature1, 0) >= target_correlations.get(feature2, 0):
                    features_to_remove.add(feature2)
                    logger.debug(f"Удаляем {feature2} (корреляция с {feature1}: {correlation:.3f})")
                else:
                    features_to_remove.add(feature1)
                    logger.debug(f"Удаляем {feature1} (корреляция с {feature2}: {correlation:.3f})")
            else:
                # Если целевая переменная не указана, удаляем второй признак по умолчанию
                features_to_remove.add(feature2)
                logger.debug(f"Удаляем {feature2} (корреляция с {feature1}: {correlation:.3f})")

        # Определяем оставшиеся признаки
        self.removed_features = list(features_to_remove)
        self.selected_features = [col for col in X.columns if col not in features_to_remove]

        logger.info(f"Выбрано {len(self.selected_features)} признаков из {X.shape[1]}. "
                   f"Удалено {len(self.removed_features)} коррелирующих признаков.")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Удаляет выбранные признаки из датафрейма.

        Args:
            X (pd.DataFrame): DataFrame с признаками.

        Returns:
            pd.DataFrame: DataFrame с выбранными признаками.

        Raises:
            ValueError: Если метод fit не был вызван.
        """
        if self.selected_features is None:
            raise ValueError("Необходимо вызвать метод fit перед transform")

        # Проверяем наличие всех выбранных признаков в датафрейме
        missing_columns = [col for col in self.selected_features if col not in X.columns]
        if missing_columns:
            logger.warning(f"В датафрейме отсутствуют следующие выбранные признаки: {missing_columns}")

        # Выбираем только те признаки, которые есть в датафрейме
        valid_columns = [col for col in self.selected_features if col in X.columns]

        return X[valid_columns]

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Выполняет корреляционный анализ и удаляет выбранные признаки.

        Args:
            X (pd.DataFrame): DataFrame с признаками.

        Returns:
            pd.DataFrame: DataFrame с выбранными признаками.
        """
        return self.fit(X).transform(X)

    def get_selected_features(self) -> Optional[List[str]]:
        """
        Возвращает список выбранных признаков.

        Returns:
            Optional[List[str]]: Список выбранных признаков или None, если fit не был вызван.
        """
        return self.selected_features

    def get_removed_features(self) -> Optional[List[str]]:
        """
        Возвращает список удаленных признаков.

        Returns:
            Optional[List[str]]: Список удаленных признаков или None, если fit не был вызван.
        """
        return self.removed_features