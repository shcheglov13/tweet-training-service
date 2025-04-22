"""
Модуль для обучения моделей с использованием FLAML.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
import joblib
import os
import json
from sklearn.base import BaseEstimator
from sklearn.utils.class_weight import compute_class_weight
from flaml import AutoML

logger = logging.getLogger('tweet_training_service.training.model_trainer')


class ModelTrainer:
    """
    Класс для обучения моделей с использованием FLAML.
    """

    def __init__(
            self,
            automl_settings: Dict[str, Any],
            class_weight: Optional[Union[str, Dict[int, float]]] = 'balanced',
            model_dir: str = 'models'
    ):
        """
        Инициализирует тренер моделей.

        Args:
            automl_settings (Dict[str, Any]): Настройки для FLAML AutoML.
            class_weight (Union[str, Dict[int, float]], optional):
                Веса классов:
                - 'balanced': автоматически вычислять веса классов
                - None: все классы имеют вес 1
                - словарь {класс: вес}: конкретные веса для каждого класса
            model_dir (str): Директория для сохранения моделей.
        """
        self.automl_settings = automl_settings
        self.class_weight = class_weight
        self.model_dir = model_dir
        self.automl = AutoML()
        self.feature_names = None

        # Создаем директорию для сохранения моделей, если она не существует
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def _compute_sample_weights(self, y: pd.Series) -> np.ndarray:
        """
        Вычисляет веса для каждого примера на основе заданной стратегии.

        Args:
            y (pd.Series): Серия с целевой переменной.

        Returns:
            np.ndarray: Массив с весами для каждого примера.
        """
        if self.class_weight is None:
            # Все примеры имеют одинаковый вес
            return np.ones(len(y))

        elif isinstance(self.class_weight, dict):
            # Используем предоставленные веса классов
            return np.array([self.class_weight.get(cls, 1.0) for cls in y])

        elif self.class_weight == 'balanced':
            # Вычисляем веса на основе распределения классов
            classes = np.unique(y)
            weights = compute_class_weight(
                class_weight='balanced',
                classes=classes,
                y=y
            )

            # Создаем словарь {класс: вес}
            class_weight_dict = {cls: weight for cls, weight in zip(classes, weights)}
            logger.info(f"Вычисленные веса классов: {class_weight_dict}")

            # Присваиваем вес каждому примеру
            return np.array([class_weight_dict[cls] for cls in y])

        else:
            raise ValueError(f"Неподдерживаемый тип весов классов: {self.class_weight}")

    def fit(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
    ) -> 'ModelTrainer':
        """
        Обучает модель на предоставленных данных.

        Args:
            X_train (pd.DataFrame): Обучающая выборка - признаки.
            y_train (pd.Series): Обучающая выборка - целевая переменная.
            X_val (pd.DataFrame, optional): Валидационная выборка - признаки.
            y_val (pd.Series, optional): Валидационная выборка - целевая переменная.

        Returns:
            ModelTrainer: Экземпляр класса для цепочки вызовов.
        """
        logger.info(f"Начало обучения модели на {len(X_train)} примерах")
        self.feature_names = list(X_train.columns)

        # Вычисляем веса примеров для обработки несбалансированных данных
        sample_weight = self._compute_sample_weights(y_train)

        # Составляем настройки для FLAML
        fit_kwargs = {'sample_weight': sample_weight}

        # Обучаем модель
        self.automl.fit(
            X_train=X_train,
            y_train=y_train,
            **self.automl_settings,
            **fit_kwargs
        )

        logger.info(f"Обучение модели завершено. Выбранный алгоритм: {self.automl.best_estimator}")

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Выполняет предсказание классов.

        Args:
            X (pd.DataFrame): DataFrame с признаками.

        Returns:
            np.ndarray: Массив с предсказанными классами.

        Raises:
            ValueError: Если модель не была обучена.
        """
        if self.automl.model is None:
            raise ValueError("Модель не была обучена. Сначала вызовите метод fit.")

        return self.automl.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Выполняет предсказание вероятностей классов.

        Args:
            X (pd.DataFrame): DataFrame с признаками.

        Returns:
            np.ndarray: Массив с предсказанными вероятностями классов.

        Raises:
            ValueError: Если модель не была обучена или не поддерживает predict_proba.
        """
        if self.automl.model is None:
            raise ValueError("Модель не была обучена. Сначала вызовите метод fit.")

        try:
            return self.automl.predict_proba(X)
        except AttributeError:
            raise ValueError("Выбранная модель не поддерживает предсказание вероятностей")

    def get_model(self) -> BaseEstimator:
        """
        Возвращает обученную модель.

        Returns:
            BaseEstimator: Обученная модель.

        Raises:
            ValueError: Если модель не была обучена.
        """
        if self.automl.model is None:
            raise ValueError("Модель не была обучена. Сначала вызовите метод fit.")

        return self.automl.model

    def get_best_config(self) -> Dict[str, Any]:
        """
        Возвращает лучшую конфигурацию модели.

        Returns:
            Dict[str, Any]: Словарь с параметрами лучшей модели.

        Raises:
            ValueError: Если модель не была обучена.
        """
        if self.automl.model is None:
            raise ValueError("Модель не была обучена. Сначала вызовите метод fit.")

        return self.automl.best_config

    def get_leaderboard(self) -> pd.DataFrame:
        """
        Возвращает таблицу лучших моделей.

        Returns:
            pd.DataFrame: Таблица с результатами обученных моделей.

        Raises:
            ValueError: Если модель не была обучена.
        """
        if self.automl.model is None:
            raise ValueError("Модель не была обучена. Сначала вызовите метод fit.")

        return self.automl.leaderboard()

    def save_model(self, model_name: str) -> str:
        """
        Сохраняет обученную модель на диск.

        Args:
            model_name (str): Базовое имя модели.

        Returns:
            str: Путь к сохраненной модели.

        Raises:
            ValueError: Если модель не была обучена.
        """
        if self.automl.model is None:
            raise ValueError("Модель не была обучена. Сначала вызовите метод fit.")

        model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
        joblib.dump(self.automl, model_path)

        # Сохраняем конфигурацию модели
        config_path = os.path.join(self.model_dir, f"{model_name}_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                'best_estimator': self.automl.best_estimator,
                'best_config': self.automl.best_config,
                'feature_names': self.feature_names
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"Модель сохранена в {model_path}")
        logger.info(f"Конфигурация модели сохранена в {config_path}")

        return model_path

    @classmethod
    def load_model(cls, model_path: str) -> 'ModelTrainer':
        """
        Загружает обученную модель с диска.

        Args:
            model_path (str): Путь к сохраненной модели.

        Returns:
            ModelTrainer: Экземпляр класса с загруженной моделью.

        Raises:
            FileNotFoundError: Если файл модели не найден.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")

        logger.info(f"Загрузка модели из {model_path}")

        # Загружаем модель
        automl = joblib.load(model_path)

        # Создаем экземпляр класса
        trainer = cls(automl_settings={}, model_dir=os.path.dirname(model_path))
        trainer.automl = automl

        # Загружаем имена признаков из конфигурации
        config_path = model_path.replace('.joblib', '_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                trainer.feature_names = config.get('feature_names', None)

        return trainer