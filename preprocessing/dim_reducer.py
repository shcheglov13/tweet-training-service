import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class DimensionalityReducer(BaseEstimator, TransformerMixin):
    """
    Трансформер для снижения размерности эмбеддингов.
    Реализует интерфейс sklearn.base.TransformerMixin для использования в пайплайне.
    """

    def __init__(self,
                 text_method: str = "pca",
                 text_n_components: int = 50,
                 quoted_text_method: str = "pca",
                 quoted_text_n_components: int = 50,
                 image_method: str = "pca",
                 image_n_components: int = 100,
                 random_state: int = 42) -> None:
        """
        Инициализирует редуктор размерности.

        Args:
            text_method: Метод снижения размерности для эмбеддингов основного текста.
            text_n_components: Количество компонент для эмбеддингов основного текста.
            quoted_text_method: Метод снижения размерности для эмбеддингов цитируемого текста.
            quoted_text_n_components: Количество компонент для эмбеддингов цитируемого текста.
            image_method: Метод снижения размерности для эмбеддингов изображений.
            image_n_components: Количество компонент для эмбеддингов изображений.
            random_state: Случайное начальное число для воспроизводимости.
        """
        self.text_method = text_method
        self.text_n_components = text_n_components
        self.quoted_text_method = quoted_text_method
        self.quoted_text_n_components = quoted_text_n_components
        self.image_method = image_method
        self.image_n_components = image_n_components
        self.random_state = random_state

        # Инициализация трансформеров
        self.text_transformer = None
        self.quoted_text_transformer = None
        self.image_transformer = None

        # Настройка логгера
        self.logger = logger

    def _create_transformer(self, method: str, n_components: int) -> BaseEstimator:
        """
        Создает трансформер для снижения размерности.

        Args:
            method: Метод снижения размерности ('pca', и т.д.).
            n_components: Количество компонент.

        Returns:
            Инициализированный трансформер.

        Raises:
            ValueError: Если указан неподдерживаемый метод.
        """
        if method.lower() == "pca":
            return PCA(n_components=n_components, random_state=self.random_state)
        else:
            raise ValueError(f"Неподдерживаемый метод снижения размерности: {method}")

    def _get_embedding_columns(self, X: pd.DataFrame, prefix: str) -> List[str]:
        """
        Получает список колонок с эмбеддингами.

        Args:
            X: Датафрейм с признаками.
            prefix: Префикс колонок с эмбеддингами.

        Returns:
            Список колонок с эмбеддингами.
        """
        return [col for col in X.columns if col.startswith(prefix)]

    def _reduce_dimensions(self,
                           X: pd.DataFrame,
                           columns: List[str],
                           transformer: BaseEstimator,
                           prefix: str) -> pd.DataFrame:
        """
        Применяет снижение размерности к указанным колонкам.

        Args:
            X: Исходный датафрейм.
            columns: Колонки с эмбеддингами.
            transformer: Трансформер для снижения размерности.
            prefix: Префикс для новых колонок.

        Returns:
            Датафрейм с новыми колонками с уменьшенными размерностями.
        """
        if not columns:
            return pd.DataFrame()

        # Извлекаем эмбеддинги
        embeddings = X[columns].values

        if transformer is None:
            self.logger.warning(f"Трансформер для {prefix} не инициализирован")
            return pd.DataFrame()

        # Применяем трансформацию
        reduced_embeddings = transformer.transform(embeddings)

        # Создаем новый датафрейм с уменьшенными размерностями
        reduced_df = pd.DataFrame(
            reduced_embeddings,
            columns=[f"{prefix}_reduced_{i}" for i in range(reduced_embeddings.shape[1])],
            index=X.index
        )

        return reduced_df

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DimensionalityReducer':
        """
        Обучает трансформеры для снижения размерности.

        Args:
            X: Датафрейм с признаками.
            y: Целевая переменная (не используется).

        Returns:
            self.
        """
        self.logger.info("Инициализация трансформеров для снижения размерности")

        # Получаем колонки с эмбеддингами
        text_emb_columns = self._get_embedding_columns(X, "text_emb_")
        quoted_text_emb_columns = self._get_embedding_columns(X, "quoted_text_emb_")
        image_emb_columns = self._get_embedding_columns(X, "image_emb_")

        # Создаем и обучаем трансформеры, если есть соответствующие колонки
        if text_emb_columns:
            self.logger.info(f"Обучение трансформера для эмбеддингов основного текста, метод: {self.text_method}")
            self.text_transformer = self._create_transformer(self.text_method, self.text_n_components)
            self.text_transformer.fit(X[text_emb_columns].values)

        if quoted_text_emb_columns:
            self.logger.info(
                f"Обучение трансформера для эмбеддингов цитируемого текста, метод: {self.quoted_text_method}")
            self.quoted_text_transformer = self._create_transformer(self.quoted_text_method,
                                                                    self.quoted_text_n_components)
            self.quoted_text_transformer.fit(X[quoted_text_emb_columns].values)

        if image_emb_columns:
            self.logger.info(f"Обучение трансформера для эмбеддингов изображений, метод: {self.image_method}")
            self.image_transformer = self._create_transformer(self.image_method, self.image_n_components)
            self.image_transformer.fit(X[image_emb_columns].values)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Преобразует эмбеддинги в уменьшенное пространство.

        Args:
            X: Датафрейм с признаками.

        Returns:
            Датафрейм с трансформированными признаками.
        """
        self.logger.info("Снижение размерности эмбеддингов")

        # Получаем колонки с эмбеддингами
        text_emb_columns = self._get_embedding_columns(X, "text_emb_")
        quoted_text_emb_columns = self._get_embedding_columns(X, "quoted_text_emb_")
        image_emb_columns = self._get_embedding_columns(X, "image_emb_")

        # Создаем копию датафрейма без колонок с эмбеддингами
        all_emb_columns = text_emb_columns + quoted_text_emb_columns + image_emb_columns
        result_df = X.drop(columns=all_emb_columns)

        # Применяем трансформации к каждому типу эмбеддингов
        reduced_dfs = []

        if text_emb_columns and self.text_transformer is not None:
            text_reduced_df = self._reduce_dimensions(X, text_emb_columns, self.text_transformer, "text")
            reduced_dfs.append(text_reduced_df)

        if quoted_text_emb_columns and self.quoted_text_transformer is not None:
            quoted_text_reduced_df = self._reduce_dimensions(X, quoted_text_emb_columns, self.quoted_text_transformer,
                                                             "quoted_text")
            reduced_dfs.append(quoted_text_reduced_df)

        if image_emb_columns and self.image_transformer is not None:
            image_reduced_df = self._reduce_dimensions(X, image_emb_columns, self.image_transformer, "image")
            reduced_dfs.append(image_reduced_df)

        # Объединяем все датафреймы
        result_df = pd.concat([result_df] + reduced_dfs, axis=1)

        self.logger.info(f"Размерность после снижения: {result_df.shape}")

        return result_df