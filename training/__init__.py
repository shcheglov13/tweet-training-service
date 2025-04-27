"""Модуль для обучения и оценки модели."""

from .pipeline import TweetPipeline
from .trainer import ModelTrainer

__all__ = ["TweetPipeline", "ModelTrainer"]