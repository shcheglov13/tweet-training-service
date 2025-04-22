"""
Модуль для визуализации данных и результатов.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
import os
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

logger = logging.getLogger('tweet_training_service.visualization.visualization')


class Visualizer:
    """
    Класс для визуализации данных и результатов обучения.
    """

    def __init__(self, output_dir: str = 'visualizations'):
        """
        Инициализирует визуализатор.

        Args:
            output_dir (str): Директория для сохранения визуализаций.
        """
        self.output_dir = output_dir

        # Создаем директорию для визуализаций, если она не существует
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Настраиваем стиль
        sns.set(style="whitegrid")

    def plot_class_distribution(
            self,
            y: pd.Series,
            title: str = 'Распределение классов',
            labels: Optional[List[str]] = None,
            filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Визуализирует распределение классов.

        Args:
            y (pd.Series): Серия с целевой переменной.
            title (str): Заголовок графика.
            labels (List[str], optional): Названия классов.
            filename (str, optional): Имя файла для сохранения графика.

        Returns:
            plt.Figure: Figure с визуализацией распределения классов.
        """
        logger.info("Визуализация распределения классов")

        # Вычисляем частоты классов
        value_counts = y.value_counts().sort_index()

        # Если метки не указаны, используем числовые значения
        if labels is None:
            labels = [f"Класс {i}" for i in value_counts.index]
        elif len(labels) != len(value_counts):
            logger.warning(f"Количество меток ({len(labels)}) не соответствует "
                           f"количеству классов ({len(value_counts)})")
            labels = [f"Класс {i}" for i in value_counts.index]

        # Создаем DataFrame для визуализации
        plot_df = pd.DataFrame({
            'Класс': [labels[i] for i in value_counts.index],
            'Количество': value_counts.values,
            'Процент': value_counts.values / len(y) * 100
        })

        # Строим график
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x='Класс', y='Количество', data=plot_df)

        # Добавляем проценты над столбцами
        for i, p in enumerate(ax.patches):
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2.,
                    height + 5,
                    f'{plot_df["Процент"].iloc[i]:.1f}%',
                    ha="center")

        plt.title(title)
        plt.ylabel("Количество примеров")
        plt.tight_layout()

        # Сохраняем график, если указано имя файла
        if filename:
            full_path = os.path.join(self.output_dir, filename)
            plt.savefig(full_path, bbox_inches='tight', dpi=300)
            logger.info(f"График распределения классов сохранен в {full_path}")

        return plt.gcf()

    def plot_feature_correlation(
            self,
            X: pd.DataFrame,
            top_n: int = 20,
            figsize: Tuple[int, int] = (12, 10),
            filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Визуализирует корреляционную матрицу признаков.

        Args:
            X (pd.DataFrame): DataFrame с признаками.
            top_n (int): Количество признаков с наибольшей корреляцией для отображения.
            figsize (Tuple[int, int]): Размер графика.
            filename (str, optional): Имя файла для сохранения графика.

        Returns:
            plt.Figure: Figure с визуализацией корреляционной матрицы.
        """
        logger.info(f"Визуализация корреляционной матрицы для {top_n} признаков")

        # Выбираем только числовые признаки
        numeric_X = X.select_dtypes(include=[np.number])

        if numeric_X.shape[1] == 0:
            logger.warning("В датафрейме отсутствуют числовые признаки")
            return None

        # Вычисляем корреляционную матрицу
        corr_matrix = numeric_X.corr()

        # Находим признаки с наибольшей корреляцией
        # Для этого преобразуем матрицу корреляции в длинный формат
        corr_long = corr_matrix.unstack().reset_index()
        corr_long.columns = ['feature1', 'feature2', 'correlation']

        # Удаляем самокорреляцию и дубликаты (corr(a,b) = corr(b,a))
        corr_long = corr_long[corr_long['feature1'] != corr_long['feature2']]
        corr_long['pair'] = corr_long.apply(
            lambda row: tuple(sorted([row['feature1'], row['feature2']])),
            axis=1
        )
        corr_long = corr_long.drop_duplicates('pair')

        # Сортируем по абсолютному значению корреляции
        corr_long['abs_correlation'] = corr_long['correlation'].abs()
        corr_long = corr_long.sort_values('abs_correlation', ascending=False)

        # Берем топ-N признаков с наибольшей корреляцией
        top_features = set()
        for _, row in corr_long.head(top_n).iterrows():
            top_features.add(row['feature1'])
            top_features.add(row['feature2'])

        top_features = list(top_features)[:top_n]

        # Создаем подматрицу для визуализации
        top_corr = corr_matrix.loc[top_features, top_features]

        # Строим тепловую карту
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(top_corr, dtype=bool))
        sns.heatmap(
            top_corr,
            mask=mask,
            cmap='coolwarm',
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=0.5,
            annot=True,
            fmt=".2f",
            cbar_kws={"shrink": .8}
        )
        plt.title(f'Корреляционная матрица топ-{len(top_features)} признаков')
        plt.tight_layout()

        # Сохраняем график, если указано имя файла
        if filename:
            full_path = os.path.join(self.output_dir, filename)
            plt.savefig(full_path, bbox_inches='tight', dpi=300)
            logger.info(f"Корреляционная матрица признаков сохранена в {full_path}")

        return plt.gcf()

    def plot_feature_distributions(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            feature_names: Optional[List[str]] = None,
            n_cols: int = 3,
            max_features: int = 12,
            figsize: Tuple[int, int] = (15, 12),
            filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Визуализирует распределения отдельных признаков по классам.

        Args:
            X (pd.DataFrame): DataFrame с признаками.
            y (pd.Series): Серия с целевой переменной.
            feature_names (List[str], optional): Имена признаков для визуализации.
                Если None, будут визуализированы первые max_features признаков.
            n_cols (int): Количество столбцов в сетке графиков.
            max_features (int): Максимальное количество признаков для визуализации.
            figsize (Tuple[int, int]): Размер фигуры.
            filename (str, optional): Имя файла для сохранения графика.

        Returns:
            plt.Figure: Figure с визуализацией распределений признаков.
        """
        logger.info("Визуализация распределений признаков по классам")

        # Выбираем только числовые признаки
        numeric_X = X.select_dtypes(include=[np.number])

        if numeric_X.shape[1] == 0:
            logger.warning("В датафрейме отсутствуют числовые признаки")
            return None

        # Если не указаны имена признаков, берем первые max_features
        if feature_names is None:
            feature_names = numeric_X.columns[:max_features].tolist()
        else:
            # Проверяем, что указанные признаки присутствуют в датафрейме
            valid_features = [f for f in feature_names if f in numeric_X.columns]
            if len(valid_features) < len(feature_names):
                logger.warning(f"Некоторые указанные признаки отсутствуют в датафрейме. "
                               f"Используем {len(valid_features)} из {len(feature_names)}")
            feature_names = valid_features[:max_features]

        # Ограничиваем количество признаков
        if len(feature_names) > max_features:
            feature_names = feature_names[:max_features]
            logger.info(f"Ограничиваем количество признаков до {max_features}")

        # Определяем количество строк в сетке
        n_rows = (len(feature_names) + n_cols - 1) // n_cols

        # Создаем DataFrame для визуализации
        plot_df = numeric_X[feature_names].copy()
        plot_df['target'] = y

        # Строим графики
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        for i, feature in enumerate(feature_names):
            if i < len(axes):
                ax = axes[i]

                # Строим распределение признака для каждого класса
                sns.histplot(
                    data=plot_df,
                    x=feature,
                    hue='target',
                    ax=ax,
                    alpha=0.6,
                    kde=True
                )

                ax.set_title(feature)

                # Удаляем легенду для всех графиков, кроме первого
                if i > 0:
                    ax.get_legend().remove()

        # Скрываем пустые графики
        for i in range(len(feature_names), len(axes)):
            axes[i].set_visible(False)

        # Добавляем общую легенду
        plt.legend(['Низкий потенциал', 'Высокий потенциал'],
                   loc='upper center',
                   bbox_to_anchor=(0.5, 0),
                   ncol=2)

        plt.tight_layout()

        # Сохраняем график, если указано имя файла
        if filename:
            full_path = os.path.join(self.output_dir, filename)
            plt.savefig(full_path, bbox_inches='tight', dpi=300)
            logger.info(f"Распределения признаков сохранены в {full_path}")

        return fig

    def compare_models(
            self,
            model_results: Dict[str, Dict[str, float]],
            metrics: Optional[List[str]] = None,
            figsize: Tuple[int, int] = (12, 8),
            filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Сравнивает различные модели по метрикам качества.

        Args:
            model_results (Dict[str, Dict[str, float]]): Словарь с результатами моделей.
                Ключи - названия моделей, значения - словари с метриками.
            metrics (List[str], optional): Список метрик для сравнения.
                Если None, используются все доступные метрики.
            figsize (Tuple[int, int]): Размер фигуры.
            filename (str, optional): Имя файла для сохранения графика.

        Returns:
            plt.Figure: Figure с визуализацией сравнения моделей.
        """
        logger.info(f"Сравнение {len(model_results)} моделей")

        # Проверяем наличие результатов
        if not model_results:
            logger.warning("Пустой словарь с результатами моделей")
            return None

        # Создаем DataFrame для визуализации
        models = []
        metrics_list = []
        values = []

        for model_name, results in model_results.items():
            # Если метрики не указаны, используем все доступные
            if metrics is None:
                current_metrics = list(results.keys())
            else:
                # Фильтруем только доступные метрики
                current_metrics = [m for m in metrics if m in results]
                if len(current_metrics) < len(metrics):
                    logger.warning(f"Некоторые указанные метрики отсутствуют в результатах модели {model_name}")

            # Добавляем данные в списки
            for metric in current_metrics:
                models.append(model_name)
                metrics_list.append(metric)
                values.append(results[metric])

        # Создаем DataFrame
        plot_df = pd.DataFrame({
            'Модель': models,
            'Метрика': metrics_list,
            'Значение': values
        })

        # Строим график
        plt.figure(figsize=figsize)
        sns.barplot(x='Метрика', y='Значение', hue='Модель', data=plot_df)
        plt.title('Сравнение моделей по метрикам качества')
        plt.ylim(0, 1)
        plt.legend(loc='lower right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Сохраняем график, если указано имя файла
        if filename:
            full_path = os.path.join(self.output_dir, filename)
            plt.savefig(full_path, bbox_inches='tight', dpi=300)
            logger.info(f"Сравнение моделей сохранено в {full_path}")

        return plt.gcf()

    def plot_learning_curves(
            self,
            automl,
            figsize: Tuple[int, int] = (12, 8),
            filename: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """
        Визуализирует кривые обучения для AutoML.

        Args:
            automl: Объект AutoML с историей обучения.
            figsize (Tuple[int, int]): Размер фигуры.
            filename (str, optional): Имя файла для сохранения графика.

        Returns:
            Optional[plt.Figure]: Figure с визуализацией кривых обучения или None,
                если история обучения недоступна.
        """
        logger.info("Визуализация кривых обучения")

        try:
            # Получаем историю обучения
            best_estimator = automl.best_estimator

            # Строим график
            plt.figure(figsize=figsize)

            for estimator, logs in automl.estimator_audit.items():
                iterations = list(range(1, len(logs) + 1))
                values = logs

                plt.plot(iterations, values, 'o-', label=estimator)

            plt.xlabel('Итерация')
            plt.ylabel('Метрика качества')
            plt.title(f'Кривые обучения (лучший алгоритм: {best_estimator})')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()

            # Сохраняем график, если указано имя файла
            if filename:
                full_path = os.path.join(self.output_dir, filename)
                plt.savefig(full_path, bbox_inches='tight', dpi=300)
                logger.info(f"Кривые обучения сохранены в {full_path}")

            return plt.gcf()

        except (AttributeError, KeyError) as e:
            logger.warning(f"Не удалось построить кривые обучения: {e}")
            return None

    def plot_roc_curves_comparison(
            self,
            models_proba: Dict[str, Tuple[pd.Series, np.ndarray]],
            figsize: Tuple[int, int] = (10, 8),
            filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Сравнивает ROC-кривые различных моделей.

        Args:
            models_proba (Dict[str, Tuple[pd.Series, np.ndarray]]): Словарь с названиями моделей
                и кортежами (y_true, y_score).
            figsize (Tuple[int, int]): Размер фигуры.
            filename (str, optional): Имя файла для сохранения графика.

        Returns:
            plt.Figure: Figure с визуализацией сравнения ROC-кривых.
        """
        logger.info(f"Сравнение ROC-кривых для {len(models_proba)} моделей")

        plt.figure(figsize=figsize)

        for model_name, (y_true, y_score) in models_proba.items():
            fpr, tpr, _ = roc_curve(y_true, y_score)
            roc_auc = roc_auc_score(y_true, y_score)

            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')

        plt.plot([0, 1], [0, 1], 'k--', label='Случайная модель')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Сравнение ROC-кривых')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Сохраняем график, если указано имя файла
        if filename:
            full_path = os.path.join(self.output_dir, filename)
            plt.savefig(full_path, bbox_inches='tight', dpi=300)
            logger.info(f"Сравнение ROC-кривых сохранено в {full_path}")

        return plt.gcf()

    def plot_pr_curves_comparison(
            self,
            models_proba: Dict[str, Tuple[pd.Series, np.ndarray]],
            figsize: Tuple[int, int] = (10, 8),
            filename: Optional[str] = None
    ) -> plt.Figure:
        """
        Сравнивает PR-кривые различных моделей.

        Args:
            models_proba (Dict[str, Tuple[pd.Series, np.ndarray]]): Словарь с названиями моделей
                и кортежами (y_true, y_score).
            figsize (Tuple[int, int]): Размер фигуры.
            filename (str, optional): Имя файла для сохранения графика.

        Returns:
            plt.Figure: Figure с визуализацией сравнения PR-кривых.
        """
        logger.info(f"Сравнение PR-кривых для {len(models_proba)} моделей")

        plt.figure(figsize=figsize)

        for model_name, (y_true, y_score) in models_proba.items():
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            avg_precision = average_precision_score(y_true, y_score)

            plt.plot(recall, precision, label=f'{model_name} (AP = {avg_precision:.4f})')

        random_baseline = sum(list(models_proba.values())[0][0]) / len(list(models_proba.values())[0][0])
        plt.axhline(y=random_baseline, linestyle='--', color='r', label='Случайная модель')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Сравнение Precision-Recall кривых')
        plt.legend(loc="lower left")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Сохраняем график, если указано имя файла
        if filename:
            full_path = os.path.join(self.output_dir, filename)
            plt.savefig(full_path, bbox_inches='tight', dpi=300)
            logger.info(f"Сравнение PR-кривых сохранено в {full_path}")

        return plt.gcf()