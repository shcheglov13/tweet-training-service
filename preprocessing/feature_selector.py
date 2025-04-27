import logging
import pandas as pd
from typing import List, Optional
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Трансформер для отбора признаков.
    Реализует интерфейс sklearn.base.TransformerMixin для использования в пайплайне.
    """

    def __init__(self, method: str = "all_features", exclude_list: Optional[List[str]] = None) -> None:
        """
        Инициализирует селектор признаков.

        Args:
            method: Метод отбора признаков ('all_features' или 'exclude_features').
            exclude_list: Список признаков для исключения (при method='exclude_features').
        """
        self.method = method
        self.exclude_list = exclude_list or []

        # Настройка логгера
        self.logger = logger

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureSelector':
        """
        Обучает трансформер (в данном случае просто сохраняет информацию).

        Args:
            X: Датафрейм с признаками.
            y: Целевая переменная (не используется).

        Returns:
            self.
        """
        self.feature_names_ = X.columns.tolist()

        if self.method == 'all_features':
            self.selected_features_ = self.feature_names_
            self.logger.info(f"Выбраны все признаки: {len(self.selected_features_)}")
        elif self.method == 'exclude_features':
            self.selected_features_ = [f for f in self.feature_names_ if f not in self.exclude_list]
            self.logger.info(
                f"Выбраны все признаки, кроме исключенных: {len(self.selected_features_)} "
                f"из {len(self.feature_names_)}"
            )
        else:
            raise ValueError(f"Неподдерживаемый метод отбора признаков: {self.method}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Преобразует датафрейм, оставляя только выбранные признаки.

        Args:
            X: Датафрейм с признаками.

        Returns:
            Датафрейм с выбранными признаками.
        """
        # Проверяем, были ли выбраны признаки на этапе fit
        if not hasattr(self, 'selected_features_'):
            self.logger.warning("Трансформер не был обучен, возвращаем все признаки")
            return X

        # Проверяем на наличие признаков, которых не было в обучающих данных
        unknown_features = [f for f in X.columns if f not in self.feature_names_]
        if unknown_features:
            self.logger.warning(f"Обнаружены новые признаки, которых не было в обучающих данных: {unknown_features}")

        # Проверяем на отсутствие признаков, которые были в обучающих данных
        missing_features = [f for f in self.selected_features_ if f not in X.columns]
        if missing_features:
            self.logger.warning(f"Отсутствуют признаки, которые были в обучающих данных: {missing_features}")

        # Выбираем только те признаки, которые есть в датафрейме
        valid_features = [f for f in self.selected_features_ if f in X.columns]

        self.logger.info(f"Отбор признаков: {len(valid_features)} из {X.shape[1]}")

        return X[valid_features]