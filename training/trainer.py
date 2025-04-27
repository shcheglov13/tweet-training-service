import logging
import pandas as pd
import numpy as np
import os
import joblib
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from flaml import AutoML
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, accuracy_score,
    confusion_matrix
)

from config.config_schema import Config
from training.pipeline import TweetPipeline

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Класс для обучения и оценки модели классификации твитов.
    """

    def __init__(self, config: Config) -> None:
        """
        Инициализирует тренер модели.

        Args:
            config: Конфигурация для сервиса.
        """
        self.config = config
        self.automl = None
        self.pipeline = None
        self.preprocessing_pipeline = None
        self.metrics = {}
        self.logger = logger

    def setup(self, tweets: List[Dict[str, Any]]) -> None:
        """
        Настраивает компоненты для обучения модели.

        Args:
            tweets: Список твитов для настройки пайплайна.
        """
        self.logger.info("Настройка компонентов для обучения модели")

        # Создаем директории для сохранения артефактов
        os.makedirs(self.config.paths.models_dir, exist_ok=True)
        os.makedirs(self.config.paths.logs_dir, exist_ok=True)

        # Настраиваем пайплайн
        self.pipeline = TweetPipeline(self.config)

        # Создаем пайплайн предобработки
        self.preprocessing_pipeline = self.pipeline.create_preprocessing_pipeline()

        # Инициализируем AutoML
        self.automl = AutoML()

        self.logger.info("Компоненты для обучения модели настроены")

    def prepare_data(self, tweets: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Подготавливает данные для обучения модели.

        Args:
            tweets: Список твитов.

        Returns:
            Кортеж из четырех элементов: X_train, X_test, y_train, y_test.
        """
        self.logger.info("Подготовка данных для обучения модели")

        # Извлекаем признаки
        X = self.pipeline.extract_features(tweets)

        # Подготавливаем целевую переменную
        y = self.pipeline.prepare_target(tweets)

        # Разделяем данные на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = self.pipeline.split_data(X, y)

        self.logger.info("Данные для обучения модели подготовлены")

        return X_train, X_test, y_train, y_test

    def train(self, X_train: pd.DataFrame, y_train: np.ndarray) -> None:
        """
        Обучает модель на подготовленных данных.

        Args:
            X_train: Обучающие данные (признаки).
            y_train: Обучающие данные (целевая переменная).
        """
        self.logger.info("Начало обучения модели")

        # Применяем предобработку к обучающей выборке
        X_train_processed = self.preprocessing_pipeline.fit_transform(X_train)

        # Настраиваем параметры AutoML
        automl_settings = self.config.automl_settings.dict()

        # Настраиваем путь к лог-файлу
        log_file_path = os.path.join(self.config.paths.logs_dir, automl_settings.get("log_file_name", "automl.log"))
        automl_settings["log_file_name"] = log_file_path

        # Обучаем модель
        self.logger.info(
            f"Запуск AutoML с бюджетом времени {automl_settings.get('time_budget')} секунд. "
            f"Метрика: {automl_settings.get('metric')}. "
            f"Оцениваемые алгоритмы: {automl_settings.get('estimator_list')}"
        )

        self.automl.fit(X_train_processed, y_train, **automl_settings)

        self.logger.info(
            f"Обучение модели завершено. Лучший алгоритм: {self.automl.best_estimator}. "
            f"Время обучения лучшей модели: {self.automl.best_config_train_time:.2f} с."
        )

    def evaluate(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, float]:
        """
        Оценивает качество модели на тестовой выборке.

        Args:
            X_test: Тестовые данные (признаки).
            y_test: Тестовые данные (целевая переменная).

        Returns:
            Словарь с метриками качества модели.
        """
        self.logger.info("Оценка качества модели на тестовой выборке")

        # Применяем предобработку к тестовой выборке
        X_test_processed = self.preprocessing_pipeline.transform(X_test)

        # Получаем предсказания модели
        y_pred = self.automl.predict(X_test_processed)
        y_pred_proba = self.automl.predict_proba(X_test_processed)[:, 1]

        # Рассчитываем метрики качества
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "pr_auc": average_precision_score(y_test, y_pred_proba)
        }

        # Сохраняем матрицу ошибок для визуализации
        self.metrics = metrics
        self.confusion_matrix = confusion_matrix(y_test, y_pred)

        # Логируем метрики
        for metric_name, metric_value in metrics.items():
            self.logger.info(f"{metric_name}: {metric_value:.4f}")

        return metrics

    def save_model(self, model_path: Optional[str] = None) -> str:
        """
        Сохраняет обученную модель и пайплайн предобработки.

        Args:
            model_path: Путь для сохранения модели. Если None, то используется путь из конфигурации.

        Returns:
            Путь к сохраненной модели.
        """
        if model_path is None:
            model_path = os.path.join(self.config.paths.models_dir, "tweet_model.joblib")

        self.logger.info(f"Сохранение модели в {model_path}")

        # Создаем полный пайплайн, включающий предобработку и модель
        full_pipeline = {
            "preprocessing": self.preprocessing_pipeline,
            "model": self.automl
        }

        # Создаем директорию, если она не существует
        Path(os.path.dirname(model_path)).mkdir(parents=True, exist_ok=True)

        # Сохраняем пайплайн
        joblib.dump(full_pipeline, model_path)

        # Сохраняем конфигурацию модели
        config_path = os.path.join(os.path.dirname(model_path), "model_config.joblib")
        model_config = {
            "best_estimator": self.automl.best_estimator,
            "best_config": self.automl.best_config,
            "metrics": self.metrics
        }
        joblib.dump(model_config, config_path)

        self.logger.info(f"Модель успешно сохранена в {model_path}")

        return model_path

    def train_and_evaluate(self, tweets: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Выполняет полный цикл обучения и оценки модели.

        Args:
            tweets: Список твитов.

        Returns:
            Словарь с метриками качества модели.
        """
        self.logger.info("Запуск полного цикла обучения и оценки модели")

        # Настраиваем компоненты
        self.setup(tweets)

        # Подготавливаем данные
        X_train, X_test, y_train, y_test = self.prepare_data(tweets)

        # Обучаем модель
        self.train(X_train, y_train)

        # Оцениваем модель
        metrics = self.evaluate(X_test, y_test)

        # Сохраняем модель
        self.save_model()

        self.logger.info("Полный цикл обучения и оценки модели завершен")

        return metrics