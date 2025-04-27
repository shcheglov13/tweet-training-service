import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional, List, Any
from config.config_schema import Config
from flaml.automl.data import get_output_from_log

logger = logging.getLogger(__name__)


class ModelVisualizer:
    """
    Класс для визуализации результатов обучения модели.
    """

    def __init__(self, config: Config) -> None:
        """
        Инициализирует визуализатор.

        Args:
            config: Конфигурация для сервиса.
        """
        self.config = config
        self.logger = logger

        # Создаем директорию для сохранения визуализаций
        os.makedirs(self.config.paths.visualizations_dir, exist_ok=True)

    def plot_confusion_matrix(self,
                              confusion_matrix: np.ndarray,
                              save_path: Optional[str] = None) -> None:
        """
        Строит и сохраняет визуализацию матрицы ошибок.

        Args:
            confusion_matrix: Матрица ошибок.
            save_path: Путь для сохранения визуализации. Если None, то используется директория из конфигурации.
        """
        self.logger.info("Построение матрицы ошибок")

        if save_path is None:
            save_path = os.path.join(self.config.paths.visualizations_dir, "confusion_matrix.png")

        # Создаем директорию, если она не существует
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)

        # Строим матрицу ошибок
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Отрицательный", "Положительный"],
            yticklabels=["Отрицательный", "Положительный"]
        )
        plt.xlabel("Предсказанный класс")
        plt.ylabel("Истинный класс")
        plt.title("Матрица ошибок")

        # Сохраняем визуализацию
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        self.logger.info(f"Матрица ошибок сохранена в {save_path}")

    def plot_learning_curve(self, log_file_path: str, save_path: Optional[str] = None) -> None:
        """
        Строит и сохраняет кривую обучения на основе логов AutoML.

        Args:
            log_file_path: Путь к лог-файлу AutoML.
            save_path: Путь для сохранения визуализации. Если None, то используется директория из конфигурации.
        """
        self.logger.info("Построение кривой обучения")

        if save_path is None:
            save_path = os.path.join(self.config.paths.visualizations_dir, "learning_curve.png")

        # Создаем директорию, если она не существует
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)

        # Извлекаем данные из лога
        try:
            time_history, best_valid_loss_history, valid_loss_history, config_history, metric_history = get_output_from_log(
                filename=log_file_path,
                time_budget=self.config.automl_settings.time_budget
            )

            # Строим кривую обучения
            plt.figure(figsize=(10, 6))
            plt.title("Кривая обучения")
            plt.xlabel("Время (сек)")

            metric_name = self.config.automl_settings.metric
            if metric_name in ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]:
                plt.ylabel(f"Валидационный {metric_name}")
                plt.step(time_history, 1 - np.array(best_valid_loss_history), where="post")
            else:
                plt.ylabel(f"Валидационная ошибка ({metric_name})")
                plt.step(time_history, best_valid_loss_history, where="post")

            # Сохраняем визуализацию
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()

            self.logger.info(f"Кривая обучения сохранена в {save_path}")

        except Exception as e:
            self.logger.error(f"Ошибка при построении кривой обучения: {e}")

    def plot_feature_importance(
            self,
            feature_names: List[str],
            feature_importances: np.ndarray,
            save_path: Optional[str] = None,
            top_n: int = 20
    ) -> None:
        """
        Строит и сохраняет график важности признаков.

        Args:
            feature_names: Названия признаков.
            feature_importances: Значения важности признаков.
            save_path: Путь для сохранения визуализации. Если None, то используется директория из конфигурации.
            top_n: Количество топовых признаков для отображения.
        """
        self.logger.info("Построение графика важности признаков")

        if save_path is None:
            save_path = os.path.join(self.config.paths.visualizations_dir, "feature_importance.png")

        # Создаем директорию, если она не существует
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)

        try:
            # Проверяем, что feature_names и feature_importances не None
            if feature_names is None or feature_importances is None:
                self.logger.warning("feature_names или feature_importances равны None. Пропускаем построение графика.")
                return

            # Проверяем, что feature_names и feature_importances являются итерируемыми объектами
            if not isinstance(feature_names, (list, np.ndarray)) or not isinstance(feature_importances,
                                                                                   (list, np.ndarray)):
                self.logger.warning(
                    f"feature_names или feature_importances не являются списками или массивами. "
                    f"Типы: {type(feature_names)}, {type(feature_importances)}. "
                    f"Пропускаем построение графика."
                )
                return

            # Проверяем, что feature_names и feature_importances имеют одинаковую длину
            if len(feature_names) != len(feature_importances):
                self.logger.warning(
                    f"Разная длина: {len(feature_names)} и {len(feature_importances)}. "
                    f"Пропускаем построение графика."
                )
                return

            # Создаем датафрейм с важностями признаков
            importance_df = pd.DataFrame({
                "feature": list(feature_names),
                "importance": list(feature_importances)
            })

            # Сортируем по убыванию важности и выбираем топ-N признаков
            importance_df = importance_df.sort_values("importance", ascending=False).head(top_n)

            # Строим график
            plt.figure(figsize=(12, 8))
            sns.barplot(data=importance_df, x="importance", y="feature", palette="viridis")
            plt.title(f"Топ-{top_n} важных признаков")
            plt.xlabel("Важность")
            plt.ylabel("Признак")

            # Сохраняем визуализацию
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()

            self.logger.info(f"График важности признаков сохранен в {save_path}")

        except Exception as e:
            self.logger.error(f"Ошибка при построении графика важности признаков: {e}")

    def plot_roc_curve(self,
                       y_true: np.ndarray,
                       y_pred_proba: np.ndarray,
                       save_path: Optional[str] = None) -> None:
        """
        Строит и сохраняет ROC-кривую.

        Args:
            y_true: Истинные значения целевой переменной.
            y_pred_proba: Вероятности принадлежности к положительному классу.
            save_path: Путь для сохранения визуализации. Если None, то используется директория из конфигурации.
        """
        self.logger.info("Построение ROC-кривой")

        from sklearn.metrics import roc_curve, auc

        if save_path is None:
            save_path = os.path.join(self.config.paths.visualizations_dir, "roc_curve.png")

        # Создаем директорию, если она не существует
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)

        # Вычисляем ROC-кривую
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Строим ROC-кривую
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC)")
        plt.legend(loc="lower right")

        # Сохраняем визуализацию
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        self.logger.info(f"ROC-кривая сохранена в {save_path}")

    def plot_pr_curve(self,
                      y_true: np.ndarray,
                      y_pred_proba: np.ndarray,
                      save_path: Optional[str] = None) -> None:
        """
        Строит и сохраняет PR-кривую (Precision-Recall).

        Args:
            y_true: Истинные значения целевой переменной.
            y_pred_proba: Вероятности принадлежности к положительному классу.
            save_path: Путь для сохранения визуализации. Если None, то используется директория из конфигурации.
        """
        self.logger.info("Построение PR-кривой")

        from sklearn.metrics import precision_recall_curve, average_precision_score

        if save_path is None:
            save_path = os.path.join(self.config.paths.visualizations_dir, "pr_curve.png")

        # Создаем директорию, если она не существует
        Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)

        # Вычисляем PR-кривую
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        average_precision = average_precision_score(y_true, y_pred_proba)

        # Строим PR-кривую
        plt.figure(figsize=(8, 8))
        plt.plot(recall, precision, color="darkorange", lw=2, label=f"PR curve (AP = {average_precision:.3f})")
        plt.axhline(y=sum(y_true) / len(y_true), color="navy", lw=2, linestyle="--", label="Базовый уровень")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="best")

        # Сохраняем визуализацию
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        self.logger.info(f"PR-кривая сохранена в {save_path}")

    def create_all_visualizations(self,
                                  trainer_results: Dict[str, Any]) -> None:
        """
        Создает все визуализации результатов обучения.

        Args:
            trainer_results: Результаты обучения модели.
        """
        self.logger.info("Создание всех визуализаций")

        # Построение матрицы ошибок
        if "confusion_matrix" in trainer_results:
            self.plot_confusion_matrix(trainer_results["confusion_matrix"])

        # Построение кривой обучения
        log_file_path = os.path.join(self.config.paths.logs_dir, self.config.automl_settings.log_file_name)
        self.plot_learning_curve(log_file_path)

        # Построение графика важности признаков
        if "feature_names" in trainer_results and "feature_importances" in trainer_results:
            self.plot_feature_importance(
                trainer_results["feature_names"],
                trainer_results["feature_importances"]
            )

        # Построение ROC и PR кривых
        if "y_true" in trainer_results and "y_pred_proba" in trainer_results:
            self.plot_roc_curve(trainer_results["y_true"], trainer_results["y_pred_proba"])
            self.plot_pr_curve(trainer_results["y_true"], trainer_results["y_pred_proba"])

        self.logger.info("Все визуализации созданы")
