import os
import sys
from typing import Dict, Any, Optional

from config.config_loader import load_config
from data.data_loader import DataLoader
from data.data_validator import DataValidator
from training.trainer import ModelTrainer
from visualization.visualizer import ModelVisualizer
from utils.logger import setup_logger


class TweetTrainingService:
    """
    Основной класс сервиса обучения моделей классификации твитов.
    """

    def __init__(self, config_path: str) -> None:
        """
        Инициализирует сервис обучения.

        Args:
            config_path: Путь к файлу конфигурации.
        """
        # Загружаем конфигурацию
        self.config = load_config(config_path)

        # Настраиваем логгер
        log_file = os.path.join(self.config.paths.logs_dir, "training_service.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.logger = setup_logger(
            log_level=self.config.tweet_features.log_level,
            log_file=log_file,
            console_output=True
        )

        # Инициализируем компоненты
        self.data_loader = DataLoader(log_level=self.config.tweet_features.log_level)
        self.data_validator = DataValidator(log_level=self.config.tweet_features.log_level)

        self.logger.info("TweetTrainingService инициализирован")

    def run(self, data_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Запускает процесс обучения модели.

        Args:
            data_path: Путь к JSON-файлу с твитами. Если None, то используется путь из конфигурации.

        Returns:
            Словарь с результатами обучения.
        """
        self.logger.info("Запуск процесса обучения модели")

        # Определяем путь к данным
        if data_path is None:
            data_path = self.config.data_path

        if data_path is None:
            raise ValueError("Не указан путь к данным. Укажите его в конфигурации или передайте в метод run().")

        # Загружаем данные
        tweets = self.data_loader.load_json(data_path)

        # Валидируем данные
        valid_tweets, invalid_tweets = self.data_validator.validate_dataset(tweets)

        # Получаем отчет о валидации
        validation_report = self.data_validator.summary_report(valid_tweets, invalid_tweets)
        self.logger.info(f"Отчет о валидации данных: {validation_report}")

        if len(valid_tweets) == 0:
            raise ValueError("После валидации не осталось валидных твитов для обучения модели.")

        # Инициализируем тренер
        trainer = ModelTrainer(self.config)

        # Обучаем и оцениваем модель
        metrics = trainer.train_and_evaluate(valid_tweets)

        # Сохраняем визуализации
        visualizer = ModelVisualizer(self.config)

        # Собираем данные для визуализации
        visualization_data = {
            "confusion_matrix": trainer.confusion_matrix,
            "feature_names": None,
            "feature_importances": None,
            "y_true": None,
            "y_pred_proba": None
        }

        # Если модель поддерживает feature_importances_, добавляем их
        if hasattr(trainer.automl.model, "estimator") and hasattr(trainer.automl.model.estimator, "feature_importances_") and hasattr(trainer.automl, "feature_names_in_"):
            visualization_data["feature_names"] = trainer.automl.feature_names_in_
            visualization_data["feature_importances"] = trainer.automl.model.estimator.feature_importances_
        elif hasattr(trainer.automl.model, "feature_importances_") and hasattr(trainer.automl, "feature_names_in_"):
            visualization_data["feature_names"] = trainer.automl.feature_names_in_
            visualization_data["feature_importances"] = trainer.automl.model.feature_importances_

        # Создаем визуализации
        visualizer.create_all_visualizations(visualization_data)

        # Формируем результаты
        results = {
            "metrics": metrics,
            "best_estimator": trainer.automl.best_estimator,
            "best_config": trainer.automl.best_config,
            "train_time": trainer.automl.best_config_train_time,
            "model_path": os.path.join(self.config.paths.models_dir, "tweet_model.joblib"),
            "validation_report": validation_report
        }

        self.logger.info("Процесс обучения модели завершен")

        return results


def train_model(config_path: str, data_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Функция для запуска процесса обучения.

    Args:
        config_path: Путь к файлу конфигурации.
        data_path: Путь к JSON-файлу с твитами. Если None, то используется путь из конфигурации.

    Returns:
        Словарь с результатами обучения.
    """
    service = TweetTrainingService(config_path)
    return service.run(data_path)


def main():
    """
    Основная точка входа при запуске приложения.
    Запускает процесс обучения модели с использованием конфигурации.
    """
    # Определяем путь к конфигурации
    config_path = "config.yaml"

    # Если указан аргумент с путем к конфигурации, используем его
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    print(f"Запуск процесса обучения модели с конфигурацией: {config_path}")

    # Запускаем процесс обучения
    try:
        results = train_model(config_path)

        # Выводим основные результаты
        print("\nОбучение модели завершено успешно!")
        print(f"Лучшая модель: {results['best_estimator']}")
        print(f"Метрики на тестовой выборке:")
        for metric_name, metric_value in results["metrics"].items():
            print(f"  {metric_name}: {metric_value:.4f}")
        print(f"Модель сохранена в: {results['model_path']}")

    except Exception as e:
        print(f"Ошибка при обучении модели: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()