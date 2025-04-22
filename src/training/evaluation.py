"""
Модуль для оценки и интерпретации моделей.
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
import os
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, average_precision_score,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from src.training.model_trainer import ModelTrainer

logger = logging.getLogger('tweet_training_service.training.evaluation')


class ModelEvaluator:
    """
    Класс для оценки и интерпретации моделей.
    """

    def __init__(self, report_dir: str = 'reports'):
        """
        Инициализирует оценщик моделей.

        Args:
            report_dir (str): Директория для сохранения отчетов и визуализаций.
        """
        self.report_dir = report_dir
        self.metrics = None
        self.feature_importance = None

        # Создаем директорию для отчетов, если она не существует
        if not os.path.exists(report_dir):
            os.makedirs(report_dir)

    def evaluate(
            self,
            model: ModelTrainer,
            X_test: pd.DataFrame,
            y_test: pd.Series,
            threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Оценивает модель с помощью различных метрик.

        Args:
            model (ModelTrainer): Обученная модель.
            X_test (pd.DataFrame): Тестовые данные - признаки.
            y_test (pd.Series): Тестовые данные - целевая переменная.
            threshold (float): Пороговое значение для бинаризации вероятностей.

        Returns:
            Dict[str, float]: Словарь с метриками качества модели.
        """
        logger.info(f"Оценка модели на {len(X_test)} примерах")

        # Получаем предсказания
        try:
            y_proba = model.predict_proba(X_test)
            y_pred_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba
            y_pred = (y_pred_proba >= threshold).astype(int)
        except (ValueError, AttributeError):
            logger.warning("Модель не поддерживает предсказание вероятностей, используем predict")
            y_pred = model.predict(X_test)
            y_pred_proba = None

        # Вычисляем метрики
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }

        # Добавляем метрики, требующие вероятности
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            metrics['average_precision'] = average_precision_score(y_test, y_pred_proba)

        # Логируем метрики
        for metric_name, metric_value in metrics.items():
            logger.info(f"{metric_name.upper()}: {metric_value:.4f}")

        self.metrics = metrics
        return metrics

    def plot_confusion_matrix(
            self,
            y_true: pd.Series,
            y_pred: np.ndarray,
            labels: Optional[List[str]] = None,
            filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Строит и визуализирует матрицу ошибок.

        Args:
            y_true (pd.Series): Истинные значения.
            y_pred (np.ndarray): Предсказанные значения.
            labels (List[str], optional): Метки классов.
            filename (str, optional): Имя файла для сохранения графика.

        Returns:
            plt.Figure: Figure с матрицей ошибок.
        """
        logger.info("Построение матрицы ошибок")

        # Вычисляем матрицу ошибок
        cm = confusion_matrix(y_true, y_pred)

        # Создаем метки классов, если не предоставлены
        if labels is None:
            unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
            labels = [f"Класс {c}" for c in unique_classes]

        # Строим график
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Предсказанные классы')
        plt.ylabel('Истинные классы')
        plt.title('Матрица ошибок')

        # Сохраняем график, если указано имя файла
        if filename:
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            logger.info(f"Матрица ошибок сохранена в {filename}")

        return plt.gcf()

    def plot_roc_curve(
            self,
            y_true: pd.Series,
            y_score: np.ndarray,
            filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Строит и визуализирует ROC-кривую.

        Args:
            y_true (pd.Series): Истинные значения.
            y_score (np.ndarray): Предсказанные вероятности положительного класса.
            filename (str, optional): Имя файла для сохранения графика.

        Returns:
            plt.Figure: Figure с ROC-кривой.
        """
        logger.info("Построение ROC-кривой")

        # Вычисляем точки для ROC-кривой
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = roc_auc_score(y_true, y_score)

        # Строим график
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC кривая (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Случайная модель')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")

        # Сохраняем график, если указано имя файла
        if filename:
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            logger.info(f"ROC-кривая сохранена в {filename}")

        return plt.gcf()

    def plot_precision_recall_curve(
            self,
            y_true: pd.Series,
            y_score: np.ndarray,
            filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Строит и визуализирует Precision-Recall кривую.

        Args:
            y_true (pd.Series): Истинные значения.
            y_score (np.ndarray): Предсказанные вероятности положительного класса.
            filename (str, optional): Имя файла для сохранения графика.

        Returns:
            plt.Figure: Figure с Precision-Recall кривой.
        """
        logger.info("Построение Precision-Recall кривой")

        # Вычисляем точки для PR-кривой
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        avg_precision = average_precision_score(y_true, y_score)

        # Строим график
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR кривая (AP = {avg_precision:.4f})')
        plt.axhline(y=sum(y_true) / len(y_true), linestyle='--', color='r', label='Случайная модель')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")

        # Сохраняем график, если указано имя файла
        if filename:
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            logger.info(f"Precision-Recall кривая сохранена в {filename}")

        return plt.gcf()

    def extract_feature_importance(
            self,
            model: ModelTrainer,
            feature_names: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Извлекает важность признаков из модели.

        Args:
            model (ModelTrainer): Обученная модель.
            feature_names (List[str], optional): Имена признаков.

        Returns:
            pd.DataFrame: DataFrame с важностью признаков.
        """
        logger.info("Извлечение важности признаков")

        try:
            # Получаем базовую модель
            estimator = model.get_model()

            # Пытаемся получить важность признаков напрямую
            if hasattr(estimator, 'feature_importances_'):
                importances = estimator.feature_importances_
            elif hasattr(estimator, 'feature_importance'):
                importances = estimator.feature_importance()
            elif hasattr(estimator, 'coef_'):
                importances = np.abs(estimator.coef_)[0]
            else:
                logger.warning("Модель не поддерживает извлечение важности признаков")
                return pd.DataFrame()

            # Если имена признаков не предоставлены, используем имена из модели
            if feature_names is None:
                feature_names = model.feature_names

            # Если всё ещё нет имен признаков, используем индексы
            if feature_names is None or len(feature_names) != len(importances):
                feature_names = [f"Feature {i}" for i in range(len(importances))]

            # Создаем DataFrame с важностью признаков
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            })

            # Сортируем по важности
            importance_df = importance_df.sort_values('importance', ascending=False)

            self.feature_importance = importance_df
            return importance_df

        except Exception as e:
            logger.error(f"Ошибка при извлечении важности признаков: {e}")
            return pd.DataFrame()

    def plot_feature_importance(
            self,
            importance_df: pd.DataFrame,
            top_n: int = 20,
            filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Визуализирует важность признаков.

        Args:
            importance_df (pd.DataFrame): DataFrame с важностью признаков.
            top_n (int): Количество самых важных признаков для отображения.
            filename (str, optional): Имя файла для сохранения графика.

        Returns:
            plt.Figure: Figure с визуализацией важности признаков.
        """
        if importance_df.empty:
            logger.warning("Пустой DataFrame с важностью признаков")
            return None

        logger.info(f"Визуализация топ-{top_n} важных признаков")

        # Берем top_n признаков
        top_features = importance_df.head(top_n).copy()

        # Строим график
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title(f'Топ-{top_n} важных признаков')
        plt.xlabel('Важность')
        plt.ylabel('Признак')
        plt.tight_layout()

        # Сохраняем график, если указано имя файла
        if filename:
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            logger.info(f"Визуализация важности признаков сохранена в {filename}")

        return plt.gcf()

    def generate_evaluation_report(
            self,
            model: ModelTrainer,
            X_test: pd.DataFrame,
            y_test: pd.Series,
            threshold: float = 0.5,
            report_prefix: str = 'evaluation'
    ) -> Dict[str, Any]:
        """
        Генерирует полный отчет об оценке модели.

        Args:
            model (ModelTrainer): Обученная модель.
            X_test (pd.DataFrame): Тестовые данные - признаки.
            y_test (pd.Series): Тестовые данные - целевая переменная.
            threshold (float): Пороговое значение для бинаризации вероятностей.
            report_prefix (str): Префикс для имен файлов отчета.

        Returns:
            Dict[str, Any]: Словарь с результатами оценки.
        """
        logger.info(f"Генерация отчета об оценке модели с префиксом '{report_prefix}'")

        # Создаем директорию для отчетов, если она не существует
        if not os.path.exists(self.report_dir):
            os.makedirs(self.report_dir)

        # Получаем предсказания
        try:
            y_proba = model.predict_proba(X_test)
            y_pred_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba
            y_pred = (y_pred_proba >= threshold).astype(int)
            has_proba = True
        except (ValueError, AttributeError):
            logger.warning("Модель не поддерживает предсказание вероятностей, используем predict")
            y_pred = model.predict(X_test)
            y_pred_proba = None
            has_proba = False

        # Оцениваем модель
        metrics = self.evaluate(model, X_test, y_test, threshold)

        # Строим матрицу ошибок
        cm_filename = os.path.join(self.report_dir, f"{report_prefix}_confusion_matrix.png")
        self.plot_confusion_matrix(
            y_test, y_pred,
            labels=["Низкий потенциал", "Высокий потенциал"],
            filename=cm_filename
        )

        # Строим ROC и PR кривые, если есть вероятности
        roc_filename = None
        pr_filename = None

        if has_proba:
            roc_filename = os.path.join(self.report_dir, f"{report_prefix}_roc_curve.png")
            self.plot_roc_curve(y_test, y_pred_proba, filename=roc_filename)

            pr_filename = os.path.join(self.report_dir, f"{report_prefix}_pr_curve.png")
            self.plot_precision_recall_curve(y_test, y_pred_proba, filename=pr_filename)

        # Извлекаем и визуализируем важность признаков
        importance_df = self.extract_feature_importance(model)
        importance_filename = None

        if not importance_df.empty:
            importance_filename = os.path.join(self.report_dir, f"{report_prefix}_feature_importance.png")
            self.plot_feature_importance(importance_df, filename=importance_filename)

            # Сохраняем важность признаков в CSV
            importance_csv = os.path.join(self.report_dir, f"{report_prefix}_feature_importance.csv")
            importance_df.to_csv(importance_csv, index=False)
            logger.info(f"Важность признаков сохранена в {importance_csv}")

        # Собираем результаты
        results = {
            'metrics': metrics,
            'confusion_matrix_path': cm_filename,
            'roc_curve_path': roc_filename,
            'pr_curve_path': pr_filename,
            'feature_importance_path': importance_filename,
            'feature_importance': importance_df
        }

        # Сохраняем результаты в JSON
        results_json = os.path.join(self.report_dir, f"{report_prefix}_results.json")
        with open(results_json, 'w', encoding='utf-8') as f:
            # Преобразуем DataFrame в словарь для JSON
            json_results = {k: (v.to_dict('records') if isinstance(v, pd.DataFrame) else v)
                            for k, v in results.items()}
            json.dump(json_results, f, ensure_ascii=False, indent=2)

        logger.info(f"Отчет об оценке модели сохранен в {results_json}")

        return results