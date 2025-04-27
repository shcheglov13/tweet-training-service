import logging
from typing import Dict, List, Any, Tuple

logger = logging.getLogger(__name__)


class DataValidator:
    """Класс для валидации данных твитов."""

    def __init__(self, log_level: str = "INFO") -> None:
        """
        Инициализирует валидатор данных.

        Args:
            log_level: Уровень логирования.
        """
        self.logger = logger

    def validate_tweet(self, tweet: Dict[str, Any]) -> bool:
        """
        Валидирует отдельный твит.

        Args:
            tweet: Словарь с данными твита.

        Returns:
            True, если твит валидный, иначе False.
        """
        # Проверка типа поста
        valid_types = {"SINGLE", "QUOTE", "RETWEET", "REPLY"}
        if "tweet_type" not in tweet or tweet["tweet_type"] not in valid_types:
            self.logger.debug(f"Невалидный тип твита: {tweet.get('tweet_type')}")
            return False

        # Проверка наличия даты создания
        if "created_at" not in tweet or not tweet["created_at"]:
            self.logger.debug(f"Отсутствует дата создания в твите: {tweet.get('id')}")
            return False

        # Проверка наличия хотя бы одного из элементов: основной текст, цитируемый текст или изображение
        has_text = "text" in tweet and tweet["text"]
        has_quoted_text = "quoted_text" in tweet and tweet["quoted_text"]
        has_image = "image_url" in tweet and tweet["image_url"]

        if not (has_text or has_quoted_text or has_image):
            self.logger.debug(f"Твит не содержит ни текста, ни цитаты, ни изображения: {tweet.get('id')}")
            return False

        return True

    def validate_dataset(self, tweets: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Валидирует датасет твитов и отсеивает невалидные записи.

        Args:
            tweets: Список словарей с данными твитов.

        Returns:
            Кортеж из двух списков: валидные твиты и невалидные твиты.
        """
        valid_tweets = []
        invalid_tweets = []

        self.logger.info(f"Начало валидации {len(tweets)} твитов")

        for tweet in tweets:
            if self.validate_tweet(tweet):
                valid_tweets.append(tweet)
            else:
                invalid_tweets.append(tweet)

        self.logger.info(
            f"Валидация завершена. Валидных твитов: {len(valid_tweets)}, невалидных: {len(invalid_tweets)}")

        return valid_tweets, invalid_tweets

    def summary_report(self, valid_tweets: List[Dict[str, Any]], invalid_tweets: List[Dict[str, Any]]) -> Dict[
        str, Any]:
        """
        Создает отчет о результатах валидации.

        Args:
            valid_tweets: Список валидных твитов.
            invalid_tweets: Список невалидных твитов.

        Returns:
            Словарь с отчетом о валидации.
        """
        report = {
            "total_tweets": len(valid_tweets) + len(invalid_tweets),
            "valid_tweets": len(valid_tweets),
            "invalid_tweets": len(invalid_tweets),
            "valid_percentage": round(len(valid_tweets) / (len(valid_tweets) + len(invalid_tweets)) * 100, 2) if (
                                                                                                                             len(valid_tweets) + len(
                                                                                                                         invalid_tweets)) > 0 else 0,
        }

        # Анализ типов валидных твитов
        tweet_types = {}
        for tweet in valid_tweets:
            tweet_type = tweet.get("tweet_type", "UNKNOWN")
            tweet_types[tweet_type] = tweet_types.get(tweet_type, 0) + 1

        report["tweet_types"] = tweet_types

        # Анализ содержимого валидных твитов
        content_stats = {
            "with_text": sum(1 for tweet in valid_tweets if tweet.get("text")),
            "with_quoted_text": sum(1 for tweet in valid_tweets if tweet.get("quoted_text")),
            "with_image": sum(1 for tweet in valid_tweets if tweet.get("image_url")),
            "with_only_text": sum(1 for tweet in valid_tweets if
                                  tweet.get("text") and not tweet.get("quoted_text") and not tweet.get("image_url")),
            "with_only_quoted_text": sum(1 for tweet in valid_tweets if
                                         not tweet.get("text") and tweet.get("quoted_text") and not tweet.get(
                                             "image_url")),
            "with_only_image": sum(1 for tweet in valid_tweets if
                                   not tweet.get("text") and not tweet.get("quoted_text") and tweet.get("image_url")),
        }

        report["content_stats"] = content_stats

        return report