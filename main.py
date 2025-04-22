"""
Основной скрипт для запуска процесса обучения модели.
"""
import os
import sys
import json
import argparse
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Добавляем текущую директорию в путь импорта
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from src.data_processing.target_binarization import TargetBinarizer
from src.data_processing.feature_selection import FeatureSelector
from src.data_processing.dataset_splitter import DatasetSplitter
from src.training.model_trainer import ModelTrainer
from src.training.evaluation import ModelEvaluator
from src.visualization.visualization import Visualizer
from src.mlflow.tracking import MLflowTracker

from tweet_features.features.feature_pipeline import FeaturePipeline
from tweet_features.config.feature_config import FeatureConfig


def parse_arguments():
    """
    Разбор аргументов командной строки.

    Returns:
        argparse.Namespace: Объект с аргументами.
    """
    parser = argparse.ArgumentParser(description='Обучение моделей классификации твитов')

    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Путь к файлу конфигурации'
    )

    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Путь к файлу с данными в формате JSON'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Уровень логирования'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Директория для сохранения результатов'
    )

    parser.add_argument(
        '--experiment-name',
        type=str,
        default='tweet-classification',
        help='Название эксперимента MLflow'
    )

    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='Название запуска MLflow'
    )

    parser.add_argument(
        '--no-mlflow',
        action='store_true',
        help='Отключить интеграцию с MLflow'
    )

    return parser.parse_args()


def setup_directories(output_dir):
    """
    Создает необходимые директории для выходных данных.

    Args:
        output_dir (str): Базовая директория для результатов.

    Returns:
        dict: Словарь с путями к директориям.
    """
    # Создаем основную директорию
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Создаем поддиректории
    dirs = {
        'models': os.path.join(output_dir, 'models'),
        'reports': os.path.join(output_dir, 'reports'),
        'visualizations': os.path.join(output_dir, 'visualizations'),
        'logs': os.path.join(output_dir, 'logs')
    }

    for dir_path in dirs.values():
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    return dirs


def load_data(data_path):
    """
    Загружает данные из JSON файла.

    Args:
        data_path (str): Путь к файлу с данными.

    Returns:
        List[dict]: Список твитов.
    """
    logger = logging.getLogger('tweet_training_service.main')
    logger.info(f"Загрузка данных из {data_path}")

    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"Загружено {len(data)} твитов")
        return data
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {e}")
        raise


def extract_features(tweets, feature_config):
    """
    Извлекает признаки из твитов.

    Args:
        tweets (List[dict]): Список твитов.
        feature_config (dict): Конфигурация для извлечения признаков.

    Returns:
        pd.DataFrame: DataFrame с признаками.
    """
    logger = logging.getLogger('tweet_training_service.main')
    logger.info("Начало извлечения признаков")

    # Создаем конфигурацию для пакета tweet_features
    config = FeatureConfig(
        use_cache=feature_config.get("use_cache", True),
        cache_dir=feature_config.get("cache_dir", "./cache"),
        device=feature_config.get("device", "cuda"),
        dim_reduction_method=feature_config.get("dim_reduction_method", "PCA"),
        text_embedding_dim=feature_config.get("text_embedding_dim", 30),
        image_embedding_dim=feature_config.get("image_embedding_dim", 60),
        batch_size=feature_config.get("batch_size", 32),
        log_level=feature_config.get("log_level", "INFO")
    )

    # Создаем пайплайн для извлечения признаков
    pipeline = FeaturePipeline(
        config=config,
        use_structural=True,
        use_text=True,
        use_image=True,
        use_emotional=True,
        use_bert_embeddings=True
    )

    # Извлекаем признаки
    features_df = pipeline.extract_to_dataframe(tweets)

    # Добавляем целевую переменную
    tx_count_values = []
    for i, tweet in enumerate(tweets):
        if 'tx_count' not in tweet:
            logger.warning(f"Твит #{i} не содержит поля 'tx_count'. Используется значение по умолчанию 0.")
            tx_count_values.append(0)
        else:
            tx_count_values.append(tweet['tx_count'])

    features_df['tx_count'] = tx_count_values

    logger.info(f"Извлечено {features_df.shape[1] - 1} признаков для {features_df.shape[0]} твитов")
    return features_df


def preprocess_data(features_df, config):
    """
    Предобрабатывает данные перед обучением.

    Args:
        features_df (pd.DataFrame): DataFrame с признаками.
        config (dict): Конфигурация предобработки.

    Returns:
        tuple: Кортеж (x_train, x_test, y_train, y_test, feature_names, binarizer, selector).
    """
    logger = logging.getLogger('tweet_training_service.main')
    logger.info("Начало предобработки данных")

    # Разделяем признаки и целевую переменную
    X = features_df.drop('tx_count', axis=1)
    y = features_df['tx_count']

    # Бинаризация целевой переменной
    threshold = config.get('target_threshold', 'median')
    binarizer = TargetBinarizer(threshold=threshold)
    y_binary = binarizer.fit_transform(y)
    logger.info(f"Бинаризация выполнена с порогом {binarizer.get_threshold()}")

    # Отбор признаков на основе корреляции
    correlation_threshold = config.get('correlation_threshold', 0.95)
    selector = FeatureSelector(
        correlation_threshold=correlation_threshold,
        target_column='tx_count'
    )
    x_selected = selector.fit_transform(X)
    selected_features = selector.get_selected_features()
    removed_features = selector.get_removed_features()

    logger.info(f"Отбор признаков: выбрано {len(selected_features)} из {X.shape[1]}")
    logger.info(f"Удалено {len(removed_features)} высококоррелированных признаков")

    # Разделение на обучающую и тестовую выборки
    test_size = config.get('test_size', 0.2)
    random_state = config.get('random_state', 42)
    stratify = config.get('stratify', True)

    splitter = DatasetSplitter(
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    x_train, x_test, y_train, y_test = splitter.split(x_selected, y_binary)

    logger.info(f"Данные разделены на обучающую ({len(x_train)} примеров) "
                f"и тестовую ({len(x_test)} примеров) выборки")

    return x_train, x_test, y_train, y_test, selected_features, binarizer, selector


def train_model(X_train, y_train, config):
    """
    Обучает модель на предоставленных данных.

    Args:
        X_train (pd.DataFrame): Обучающая выборка - признаки.
        y_train (pd.Series): Обучающая выборка - целевая переменная.
        config (dict): Конфигурация модели.

    Returns:
        ModelTrainer: Обученная модель.
    """
    logger = logging.getLogger('tweet_training_service.main')
    logger.info("Начало обучения модели")

    # Получаем настройки для FLAML
    automl_settings = config.get('automl_settings', {})

    # Создаем тренер моделей
    class_weight = config.get('class_weight', 'balanced')
    model_dir = config.get('model_dir', 'models')

    trainer = ModelTrainer(
        automl_settings=automl_settings,
        class_weight=class_weight,
        model_dir=model_dir
    )

    # Обучаем модель
    trainer.fit(X_train, y_train)

    logger.info(f"Обучение модели завершено. "
                f"Лучший алгоритм: {trainer.automl.best_estimator}")

    return trainer


def evaluate_model(model, X_test, y_test, config, output_dirs):
    """
    Оценивает модель и генерирует отчеты.

    Args:
        model (ModelTrainer): Обученная модель.
        X_test (pd.DataFrame): Тестовая выборка - признаки.
        y_test (pd.Series): Тестовая выборка - целевая переменная.
        config (dict): Конфигурация оценки.
        output_dirs (dict): Словарь с директориями для выходных данных.

    Returns:
        dict: Результаты оценки.
    """
    logger = logging.getLogger('tweet_training_service.main')
    logger.info("Начало оценки модели")

    # Создаем оценщик моделей
    evaluator = ModelEvaluator(report_dir=output_dirs['reports'])

    # Генерируем отчет об оценке
    threshold = config.get('prediction_threshold', 0.5)
    results = evaluator.generate_evaluation_report(
        model=model,
        X_test=X_test,
        y_test=y_test,
        threshold=threshold,
        report_prefix='model_evaluation'
    )

    logger.info("Оценка модели завершена")
    return results


def visualize_results(X, y, y_binary, results, output_dirs):
    """
    Создает визуализации данных и результатов.

    Args:
        X (pd.DataFrame): DataFrame с признаками.
        y (pd.Series): Исходная целевая переменная.
        y_binary (pd.Series): Бинаризованная целевая переменная.
        results (dict): Результаты оценки модели.
        output_dirs (dict): Словарь с директориями для выходных данных.

    Returns:
        dict: Пути к созданным визуализациям.
    """
    logger = logging.getLogger('tweet_training_service.main')
    logger.info("Создание визуализаций")

    # Создаем визуализатор
    visualizer = Visualizer(output_dir=output_dirs['visualizations'])

    # Визуализируем распределение классов
    class_dist_path = 'class_distribution.png'
    visualizer.plot_class_distribution(
        y_binary,
        title='Распределение классов (бинаризованное)',
        labels=['Низкий потенциал', 'Высокий потенциал'],
        filename=class_dist_path
    )

    # Визуализируем корреляционную матрицу
    corr_path = 'feature_correlation.png'
    visualizer.plot_feature_correlation(
        X,
        top_n=20,
        figsize=(14, 12),
        filename=corr_path
    )

    # Визуализируем распределения признаков
    if 'feature_importance' in results and not results['feature_importance'].empty:
        # Берем топ-10 важных признаков
        top_features = results['feature_importance']['feature'].tolist()[:10]

        feature_dist_path = 'feature_distributions.png'
        visualizer.plot_feature_distributions(
            X,
            y_binary,
            feature_names=top_features,
            n_cols=2,
            max_features=10,
            figsize=(14, 18),
            filename=feature_dist_path
        )

    logger.info("Визуализации созданы")

    return {
        'class_distribution': os.path.join(output_dirs['visualizations'], class_dist_path),
        'feature_correlation': os.path.join(output_dirs['visualizations'], corr_path),
        'feature_distributions': os.path.join(output_dirs['visualizations'], feature_dist_path)
        if 'feature_importance' in results and not results['feature_importance'].empty
        else None
    }


def log_to_mlflow(
        mlflow_tracker,
        config,
        features_df,
        x_train,
        x_test,
        y_train,
        y_test,
        model,
        evaluation_results,
        visualization_paths,
        output_dirs
):
    """
    Логирует результаты обучения в MLflow.

    Args:
        mlflow_tracker (MLflowTracker): Трекер MLflow.
        config (dict): Конфигурация эксперимента.
        features_df (pd.DataFrame): DataFrame с признаками.
        x_train (pd.DataFrame): Обучающая выборка - признаки.
        x_test (pd.DataFrame): Тестовая выборка - признаки.
        y_train (pd.Series): Обучающая выборка - целевая переменная.
        y_test (pd.Series): Тестовая выборка - целевая переменная.
        model (ModelTrainer): Обученная модель.
        evaluation_results (dict): Результаты оценки модели.
        visualization_paths (dict): Пути к созданным визуализациям.
        output_dirs (dict): Словарь с директориями для выходных данных.
    """
    logger = logging.getLogger('tweet_training_service.main')
    logger.info("Логирование результатов в MLflow")

    # Начинаем новый запуск MLflow
    tags = {
        'model_type': model.automl.best_estimator,
        'dataset_size': str(len(features_df)),
        'features_count': str(x_train.shape[1]),
        'version': '1.0.0'
    }

    mlflow_tracker.start_run(tags=tags)

    try:
        # Логируем параметры
        params = {
            'test_size': config.get('test_size', 0.2),
            'random_state': config.get('random_state', 42),
            'target_threshold': config.get('target_threshold', 'median'),
            'correlation_threshold': config.get('correlation_threshold', 0.95),
            'class_weight': config.get('class_weight', 'balanced'),
            'prediction_threshold': config.get('prediction_threshold', 0.5),
            'best_estimator': model.automl.best_estimator,
            'train_size': len(x_train),
            'test_size_actual': len(x_test),
            'positive_class_ratio_train': y_train.mean(),
            'positive_class_ratio_test': y_test.mean()
        }

        # Добавляем параметры лучшей модели
        best_config = model.get_best_config()
        for key, value in best_config.items():
            params[f'best_config_{key}'] = value

        mlflow_tracker.log_params(params)

        # Логируем метрики
        metrics = evaluation_results['metrics']
        mlflow_tracker.log_metrics(metrics)

        # Логируем модель
        mlflow_tracker.log_model(
            model.automl,
            artifact_path="model",
            registered_model_name="tweet_classification_model"
        )

        # Логируем артефакты
        if 'confusion_matrix_path' in evaluation_results:
            mlflow_tracker.log_artifact(evaluation_results['confusion_matrix_path'])

        if 'roc_curve_path' in evaluation_results:
            mlflow_tracker.log_artifact(evaluation_results['roc_curve_path'])

        if 'pr_curve_path' in evaluation_results:
            mlflow_tracker.log_artifact(evaluation_results['pr_curve_path'])

        if 'feature_importance_path' in evaluation_results:
            mlflow_tracker.log_artifact(evaluation_results['feature_importance_path'])

        # Логируем визуализации
        for path_name, path in visualization_paths.items():
            if path and os.path.exists(path):
                mlflow_tracker.log_artifact(path)

        # Логируем лидерборд моделей
        leaderboard = model.get_leaderboard()
        leaderboard_path = os.path.join(output_dirs['reports'], 'leaderboard.csv')
        leaderboard.to_csv(leaderboard_path, index=False)
        mlflow_tracker.log_artifact(leaderboard_path)

        logger.info("Результаты успешно залогированы в MLflow")
    finally:
        # Завершаем запуск MLflow
        mlflow_tracker.end_run()


def main():
    """
    Основная функция скрипта.
    """
    # Разбор аргументов командной строки
    args = parse_arguments()

    # Создаем директории для выходных данных
    output_dirs = setup_directories(args.output_dir)

    # Настраиваем логирование
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dirs['logs'], f"training_{timestamp}.log")
    logger = setup_logger(
        'tweet_training_service',
        log_level=args.log_level,
        log_file=log_file
    )

    logger.info("Запуск процесса обучения модели")
    logger.info(f"Аргументы командной строки: {args}")

    try:
        # Загружаем конфигурацию
        config = ConfigLoader.load_config(args.config)
        logger.info(f"Конфигурация загружена из {args.config}")

        # Инициализируем трекер MLflow, если не отключен
        mlflow_tracker = None
        if not args.no_mlflow:
            mlflow_tracker = MLflowTracker(
                experiment_name=args.experiment_name,
                artifact_dir=os.path.join(args.output_dir, 'mlflow-artifacts')
            )
            logger.info(f"Инициализирован MLflow трекер для эксперимента '{args.experiment_name}'")

        # Загружаем данные
        tweets = load_data(args.data)

        # Извлекаем признаки
        features_df = extract_features(tweets, config.get('feature_extraction', {}))

        # Сохраняем извлеченные признаки
        features_path = os.path.join(output_dirs['reports'], 'extracted_features.csv')
        features_df.to_csv(features_path, index=False)
        logger.info(f"Извлеченные признаки сохранены в {features_path}")

        # Предобрабатываем данные
        x_train, x_test, y_train, y_test, selected_features, binarizer, selector = preprocess_data(
            features_df, config.get('preprocessing', {})
        )

        # Обучаем модель
        model = train_model(x_train, y_train, config.get('training', {}))

        # Сохраняем модель
        model_path = model.save_model("tweet_classification_model")
        logger.info(f"Модель сохранена в {model_path}")

        # Оцениваем модель
        evaluation_results = evaluate_model(
            model, x_test, y_test,
            config.get('evaluation', {}),
            output_dirs
        )

        # Создаем визуализации
        visualization_paths = visualize_results(
            features_df.drop('tx_count', axis=1),
            features_df['tx_count'],
            y_train.append(y_test),
            evaluation_results,
            output_dirs
        )

        # Логируем результаты в MLflow, если не отключен
        if mlflow_tracker is not None:
            log_to_mlflow(
                mlflow_tracker,
                config,
                features_df,
                x_train,
                x_test,
                y_train,
                y_test,
                model,
                evaluation_results,
                visualization_paths,
                output_dirs
            )

        logger.info("Процесс обучения модели успешно завершен")

    except Exception as e:
        logger.exception(f"Ошибка при обучении модели: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
