import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from tweet_features import FeatureConfig

from tweet_features.features.feature_pipeline import FeaturePipeline
from preprocessing.dim_reducer import DimensionalityReducer
from preprocessing.feature_selector import FeatureSelector
from config.config_schema import Config

logger = logging.getLogger(__name__)


class TweetPipeline:
    """
    Класс для создания и управления пайплайном обработки твитов.
    """

    def __init__(self, config: Config) -> None:
        """
        Инициализирует пайплайн обработки твитов.

        Args:
            config: Конфигурация для сервиса.
        """
        self.config = config
        self.pipeline = None
        self.feature_pipeline = None
        self.logger = logger

    def setup_feature_extraction(self) -> None:
        """
        Настраивает пайплайн извлечения признаков из tweet-features.
        """
        self.logger.info("Настройка пайплайна извлечения признаков")

        # Создаем объект конфигурации для tweet-features
        tweet_features_config = FeatureConfig(
            batch_size=self.config.tweet_features.batch_size,
            use_cache=self.config.tweet_features.use_cache,
            cache_dir=self.config.tweet_features.cache_dir,
            device=self.config.tweet_features.device,
            log_level=self.config.tweet_features.log_level
        )

        # Инициализируем пайплайн извлечения признаков из tweet-features
        self.feature_pipeline = FeaturePipeline(
            config=tweet_features_config,
            use_structural=True,
            use_text=True,
            use_image=True,
            use_emotional=True,
            use_bert_embeddings=True
        )

        self.logger.info("Пайплайн извлечения признаков настроен")

    def create_preprocessing_pipeline(self) -> Pipeline:
        """
        Создает пайплайн предобработки данных.

        Returns:
            Пайплайн scikit-learn для предобработки данных.
        """
        self.logger.info("Создание пайплайна предобработки данных")

        # Настраиваем снижение размерности
        dim_reducer = DimensionalityReducer(
            text_method=self.config.dimensionality_reduction.text_embeddings.method,
            text_n_components=self.config.dimensionality_reduction.text_embeddings.n_components,
            quoted_text_method=self.config.dimensionality_reduction.quoted_text_embeddings.method,
            quoted_text_n_components=self.config.dimensionality_reduction.quoted_text_embeddings.n_components,
            image_method=self.config.dimensionality_reduction.image_embeddings.method,
            image_n_components=self.config.dimensionality_reduction.image_embeddings.n_components,
            random_state=self.config.data_split.random_state
        )

        # Настраиваем отбор признаков
        feature_selector = FeatureSelector(
            method=self.config.feature_selection.method,
            exclude_list=self.config.feature_selection.exclude_list
        )

        # Создаем пайплайн предобработки
        preprocessing_pipeline = Pipeline([
            ('dimensionality_reduction', dim_reducer),
            ('feature_selection', feature_selector)
        ])

        self.logger.info("Пайплайн предобработки данных создан")

        return preprocessing_pipeline

    def extract_features(self, tweets: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Извлекает признаки из твитов с помощью tweet-features.

        Args:
            tweets: Список твитов.

        Returns:
            Датафрейм с извлеченными признаками.
        """
        self.logger.info(f"Извлечение признаков из {len(tweets)} твитов")

        if self.feature_pipeline is None:
            self.setup_feature_extraction()

        # Извлекаем признаки
        features_df = self.feature_pipeline.extract_to_dataframe(tweets)

        self.logger.info(f"Извлечено {features_df.shape[1]} признаков")

        return features_df

    def prepare_target(self, tweets: List[Dict[str, Any]]) -> np.ndarray:
        """
        Подготавливает целевую переменную.

        Args:
            tweets: Список твитов.

        Returns:
            Массив с бинаризованной целевой переменной.
        """
        self.logger.info("Подготовка целевой переменной")

        # Получаем значения tx_count
        tx_counts = [tweet.get("tx_count", 0) for tweet in tweets]

        # Бинаризуем целевую переменную
        threshold = self.config.target.threshold
        target = np.array([1 if tx_count >= threshold else 0 for tx_count in tx_counts])

        positive_count = np.sum(target)
        negative_count = len(target) - positive_count

        self.logger.info(
            f"Целевая переменная подготовлена. Порог: {threshold}. "
            f"Положительный класс: {positive_count} ({positive_count / len(target) * 100:.2f}%). "
            f"Отрицательный класс: {negative_count} ({negative_count / len(target) * 100:.2f}%)."
        )

        return target

    def split_data(self,
                   X: pd.DataFrame,
                   y: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Разделяет данные на обучающую и тестовую выборки.

        Args:
            X: Датафрейм с признаками.
            y: Массив с целевой переменной.

        Returns:
            Кортеж из четырех элементов: X_train, X_test, y_train, y_test.
        """
        self.logger.info("Разделение данных на обучающую и тестовую выборки")

        # Разделяем данные с использованием стратификации
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.data_split.test_size,
            random_state=self.config.data_split.random_state,
            stratify=y
        )

        self.logger.info(
            f"Данные разделены. Обучающая выборка: {X_train.shape[0]} примеров. "
            f"Тестовая выборка: {X_test.shape[0]} примеров."
        )

        return X_train, X_test, y_train, y_test