"""
Модуль для интеграции с MLflow.
"""
import os
import logging
import json
from typing import Dict, Any, Optional, List, Tuple, Union
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from flaml import AutoML

logger = logging.getLogger('tweet_training_service.mlflow_integration.tracking')


class MLflowTracker:
    """
    Класс для интеграции с MLflow.
    """

    def __init__(
            self,
            tracking_uri: Optional[str] = None,
            experiment_name: str = 'tweet-classification',
            artifact_dir: str = 'mlflow-artifacts'
    ):
        """
        Инициализирует трекер MLflow.

        Args:
            tracking_uri (str, optional): URI для сервера MLflow.
                Если None, используется локальный файловый URI.
            experiment_name (str): Название эксперимента.
            artifact_dir (str): Директория для артефактов MLflow.
        """
        # Настраиваем URI для отслеживания
        if tracking_uri is None:
            path = os.path.abspath(artifact_dir).replace('\\', '/')
            tracking_uri = f"file:///{path}"

        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

        # Создаем или получаем эксперимент
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Создан новый эксперимент '{experiment_name}' с ID: {experiment_id}")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Использован существующий эксперимент '{experiment_name}' с ID: {experiment_id}")
        except Exception as e:
            logger.error(f"Ошибка при настройке эксперимента: {e}")
            raise

        self.experiment_id = experiment_id
        self.experiment_name = experiment_name
        self.run_id = None
        self.client = MlflowClient()

    def start_run(
            self,
            run_name: Optional[str] = None,
            tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Начинает новый запуск MLflow.

        Args:
            run_name (str, optional): Название запуска.
            tags (Dict[str, str], optional): Теги для запуска.

        Returns:
            str: ID запуска.
        """
        mlflow.start_run(
            experiment_id=self.experiment_id,
            run_name=run_name
        )

        self.run_id = mlflow.active_run().info.run_id

        # Добавляем теги
        if tags:
            for key, value in tags.items():
                mlflow.set_tag(key, value)

        logger.info(f"Начат новый запуск MLflow с ID: {self.run_id}")
        return self.run_id

    def end_run(self) -> None:
        """
        Завершает текущий запуск MLflow.
        """
        if mlflow.active_run():
            mlflow.end_run()
            logger.info(f"Завершен запуск MLflow с ID: {self.run_id}")
            self.run_id = None

    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Логирует параметры.

        Args:
            params (Dict[str, Any]): Словарь с параметрами.
        """
        if not mlflow.active_run():
            logger.warning("Нет активного запуска MLflow. Параметры не будут залогированы.")
            return

        # Преобразуем сложные типы данных в строки
        processed_params = {}
        for key, value in params.items():
            if isinstance(value, (dict, list, tuple)):
                processed_params[key] = json.dumps(value)
            else:
                processed_params[key] = value

        mlflow.log_params(processed_params)
        logger.info(f"Залогировано {len(processed_params)} параметров")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Логирует метрики.

        Args:
            metrics (Dict[str, float]): Словарь с метриками.
            step (int, optional): Шаг, на котором логируются метрики.
        """
        if not mlflow.active_run():
            logger.warning("Нет активного запуска MLflow. Метрики не будут залогированы.")
            return

        mlflow.log_metrics(metrics, step=step)
        logger.info(f"Залогировано {len(metrics)} метрик" + (f" на шаге {step}" if step is not None else ""))

    def log_artifact(self, local_path: str) -> None:
        """
        Логирует артефакт.

        Args:
            local_path (str): Путь к артефакту.
        """
        if not mlflow.active_run():
            logger.warning("Нет активного запуска MLflow. Артефакт не будет залогирован.")
            return

        if not os.path.exists(local_path):
            logger.warning(f"Файл артефакта не существует: {local_path}")
            return

        mlflow.log_artifact(local_path)
        logger.info(f"Залогирован артефакт: {local_path}")

    def log_artifacts(self, local_dir: str) -> None:
        """
        Логирует все артефакты из директории.

        Args:
            local_dir (str): Директория с артефактами.
        """
        if not mlflow.active_run():
            logger.warning("Нет активного запуска MLflow. Артефакты не будут залогированы.")
            return

        if not os.path.exists(local_dir):
            logger.warning(f"Директория с артефактами не существует: {local_dir}")
            return

        mlflow.log_artifacts(local_dir)
        logger.info(f"Залогированы артефакты из директории: {local_dir}")

    def log_figure(self, figure, artifact_path: str) -> None:
        """
        Логирует matplotlib фигуру как артефакт.

        Args:
            figure: Объект matplotlib Figure.
            artifact_path (str): Путь для сохранения артефакта.
        """
        if not mlflow.active_run():
            logger.warning("Нет активного запуска MLflow. Фигура не будет залогирована.")
            return

        mlflow.log_figure(figure, artifact_path)
        logger.info(f"Залогирована фигура: {artifact_path}")

    def log_model(
            self,
            model: Any,
            artifact_path: str = "model",
            registered_model_name: Optional[str] = None
    ) -> None:
        """
        Логирует модель.

        Args:
            model (Any): Модель для логирования.
            artifact_path (str): Путь для сохранения модели.
            registered_model_name (str, optional): Имя для регистрации модели.
        """
        if not mlflow.active_run():
            logger.warning("Нет активного запуска MLflow. Модель не будет залогирована.")
            return

        # Проверяем, является ли модель экземпляром AutoML
        if isinstance(model, AutoML):
            # Для FLAML AutoML мы логируем базовую модель
            base_model = model.model
            mlflow.sklearn.log_model(
                base_model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name
            )

            # Дополнительно логируем конфигурацию AutoML
            with open("automl_config.json", "w") as f:
                json.dump({
                    "best_estimator": model.best_estimator,
                    "best_config": model.best_config,
                    "best_iteration": model.best_iteration
                }, f, indent=2)

            mlflow.log_artifact("automl_config.json")
            os.remove("automl_config.json")
        else:
            # Для обычных scikit-learn моделей
            mlflow.sklearn.log_model(
                model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name
            )

        logger.info(f"Залогирована модель: {artifact_path}" +
                    (f", зарегистрирована как {registered_model_name}" if registered_model_name else ""))

    def log_dataframe(
            self,
            df: pd.DataFrame,
            artifact_path: str,
            format: str = "csv"
    ) -> None:
        """
        Логирует DataFrame как артефакт.

        Args:
            df (pd.DataFrame): DataFrame для логирования.
            artifact_path (str): Путь для сохранения артефакта.
            format (str): Формат сохранения ('csv' или 'parquet').
        """
        if not mlflow.active_run():
            logger.warning("Нет активного запуска MLflow. DataFrame не будет залогирован.")
            return

        # Сохраняем временный файл
        temp_path = f"temp_{os.path.basename(artifact_path)}.{format}"

        try:
            if format.lower() == 'csv':
                df.to_csv(temp_path, index=False)
            elif format.lower() == 'parquet':
                df.to_parquet(temp_path, index=False)
            else:
                logger.warning(f"Неподдерживаемый формат: {format}. Используем CSV.")
                df.to_csv(temp_path, index=False)
                format = 'csv'

            # Логируем файл
            mlflow.log_artifact(temp_path, os.path.dirname(artifact_path))
            logger.info(f"Залогирован DataFrame: {artifact_path}.{format}")
        finally:
            # Удаляем временный файл
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def set_tags(self, tags: Dict[str, str]) -> None:
        """
        Устанавливает теги для текущего запуска.

        Args:
            tags (Dict[str, str]): Словарь с тегами.
        """
        if not mlflow.active_run():
            logger.warning("Нет активного запуска MLflow. Теги не будут установлены.")
            return

        for key, value in tags.items():
            mlflow.set_tag(key, value)

        logger.info(f"Установлено {len(tags)} тегов")

    def search_runs(
            self,
            filter_string: str = "",
            max_results: int = 10,
            order_by: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Ищет запуски по заданным критериям.

        Args:
            filter_string (str): Строка фильтрации.
            max_results (int): Максимальное количество результатов.
            order_by (List[str], optional): Список колонок для сортировки.

        Returns:
            pd.DataFrame: DataFrame с найденными запусками.
        """
        if order_by is None:
            order_by = ["metrics.accuracy DESC"]

        runs = mlflow.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by
        )

        logger.info(f"Найдено {len(runs)} запусков")
        return runs

    def get_best_run(
            self,
            metric_name: str = "accuracy",
            ascending: bool = False
    ) -> Optional[str]:
        """
        Возвращает ID лучшего запуска по заданной метрике.

        Args:
            metric_name (str): Имя метрики для поиска.
            ascending (bool): Сортировать по возрастанию (True) или убыванию (False).

        Returns:
            Optional[str]: ID лучшего запуска или None, если запуски не найдены.
        """
        order_direction = "ASC" if ascending else "DESC"
        runs = self.search_runs(
            filter_string=f"metrics.{metric_name} IS NOT NULL",
            max_results=1,
            order_by=[f"metrics.{metric_name} {order_direction}"]
        )

        if len(runs) == 0:
            logger.warning(f"Не найдено запусков с метрикой {metric_name}")
            return None

        best_run_id = runs.iloc[0]["run_id"]
        logger.info(f"Найден лучший запуск с ID: {best_run_id}")
        return best_run_id

    def load_model(self, run_id: str, model_path: str = "model") -> Any:
        """
        Загружает модель из запуска MLflow.

        Args:
            run_id (str): ID запуска.
            model_path (str): Путь к модели в артефактах.

        Returns:
            Any: Загруженная модель.

        Raises:
            Exception: Если модель не найдена.
        """
        try:
            model_uri = f"runs:/{run_id}/{model_path}"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Загружена модель из запуска {run_id}")
            return model
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели из запуска {run_id}: {e}")
            raise