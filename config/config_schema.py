from typing import List, Optional
from pydantic import BaseModel, Field


class TweetFeaturesConfig(BaseModel):
    """Конфигурация для пакета tweet-features."""

    use_cache: bool = True
    cache_dir: str = "./cache"
    device: str = "cuda"
    batch_size: int = 32
    log_level: str = "INFO"


class DataSplitConfig(BaseModel):
    """Конфигурация для разделения данных."""

    test_size: float = 0.2
    random_state: int = 42


class TargetConfig(BaseModel):
    """Конфигурация для бинаризации целевой переменной."""

    threshold: int = 100


class PathsConfig(BaseModel):
    """Конфигурация путей для сохранения артефактов."""

    models_dir: str = "./models"
    logs_dir: str = "./logs"
    visualizations_dir: str = "./visualizations"


class DimensionalityReductionConfig(BaseModel):
    """Конфигурация для снижения размерности."""

    method: str = "pca"
    n_components: int = 50


class DimensionalityReductionByTypeConfig(BaseModel):
    """Конфигурация для снижения размерности по типам эмбеддингов."""

    text_embeddings: DimensionalityReductionConfig = Field(default_factory=DimensionalityReductionConfig)
    quoted_text_embeddings: DimensionalityReductionConfig = Field(default_factory=DimensionalityReductionConfig)
    image_embeddings: DimensionalityReductionConfig = Field(
        default_factory=lambda: DimensionalityReductionConfig(n_components=100)
    )


class FeatureSelectionConfig(BaseModel):
    """Конфигурация для отбора признаков."""

    method: str = "exclude_features"  # или "all_features"
    exclude_list: List[str] = Field(default_factory=list)


class AutoMLSettings(BaseModel):
    """Настройки для AutoML FLAML."""

    time_budget: int = 3600
    metric: str = "average_precision"
    task: str = "classification"
    log_file_name: str = "automl.log"
    estimator_list: List[str] = Field(default_factory=lambda: ["lgbm", "xgboost", "catboost", "rf", "extra_tree"])
    n_jobs: int = -1
    ensemble: bool = True
    eval_method: str = "cv"
    n_splits: int = 5
    verbose: int = 1
    seed: int = 42


class Config(BaseModel):
    """Полная конфигурация для tweet-training-service."""

    tweet_features: TweetFeaturesConfig = Field(default_factory=TweetFeaturesConfig)
    data_split: DataSplitConfig = Field(default_factory=DataSplitConfig)
    target: TargetConfig = Field(default_factory=TargetConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    dimensionality_reduction: DimensionalityReductionByTypeConfig = Field(
        default_factory=DimensionalityReductionByTypeConfig)
    feature_selection: FeatureSelectionConfig = Field(default_factory=FeatureSelectionConfig)
    automl_settings: AutoMLSettings = Field(default_factory=AutoMLSettings)
    data_path: Optional[str] = None