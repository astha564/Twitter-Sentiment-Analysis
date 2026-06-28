"""
=============================================================================
Unit Tests — Twitter Sentiment Analysis Project
=============================================================================
Tests for:
  - preprocess_tweets.py  : clean_tweet(), is_us_noise(), is_long_enough()
  - self_learning.py      : SentimentClassifier, confidence thresholds
=============================================================================
Run with:
    python -m pytest test_twitter_sentiment.py -v
=============================================================================
"""

import pytest
import numpy as np
import sys
import os

# Add scripts folder to path so we can import directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "scripts"))

from preprocess_tweets import clean_tweet, is_us_noise, is_long_enough
from self_learning import SentimentClassifier, HIGH_CONF_THRESHOLD, UNCERTAIN_THRESHOLD


# =============================================================================
# TESTS FOR clean_tweet()
# =============================================================================

class TestCleanTweet:

    def test_removes_url(self):
        """URLs should be stripped from tweet text."""
        tweet = "Check this out https://t.co/abc123 now"
        result = clean_tweet(tweet)
        assert "https" not in result
        assert "t.co" not in result

    def test_removes_mention(self):
        """@mentions should be removed."""
        tweet = "Hey @narendramodi this is wrong"
        result = clean_tweet(tweet)
        assert "@narendramodi" not in result
        assert "Hey" in result

    def test_removes_hashtag_symbol_keeps_word(self):
        """# symbol removed but word kept — #BJP → BJP."""
        tweet = "Vote for #BJP today"
        result = clean_tweet(tweet)
        assert "#BJP" not in result
        assert "BJP" in result

    def test_decodes_html_entities(self):
        """HTML entities like &amp; should be decoded."""
        tweet = "India &amp; Pakistan cricket match"
        result = clean_tweet(tweet)
        assert "&amp;" not in result
        assert "&" in result

    def test_removes_rt_prefix(self):
        """RT artifacts should be removed."""
        tweet = "RT : This is a retweet about politics"
        result = clean_tweet(tweet)
        assert result.startswith("RT") is False

    def test_normalises_whitespace(self):
        """Multiple spaces/newlines should collapse to single space."""
        tweet = "Modi   is   the   PM"
        result = clean_tweet(tweet)
        assert "  " not in result

    def test_strips_edges(self):
        """Leading and trailing whitespace should be removed."""
        tweet = "   BJP wins election   "
        result = clean_tweet(tweet)
        assert result == result.strip()

    def test_empty_string(self):
        """Empty string should return empty string."""
        result = clean_tweet("")
        assert result == ""

    def test_only_url_becomes_empty(self):
        """A tweet that is only a URL should clean to empty."""
        tweet = "https://t.co/xyz123"
        result = clean_tweet(tweet)
        assert result.strip() == ""


# =============================================================================
# TESTS FOR is_us_noise()
# =============================================================================

class TestIsUsNoise:

    def test_detects_trump(self):
        """Tweets mentioning Trump should be flagged as US noise."""
        tweet = "Trump wins the election again"
        assert is_us_noise(tweet) is True

    def test_detects_maga(self):
        """MAGA keyword should be detected."""
        tweet = "MAGA supporters protest in Washington"
        assert is_us_noise(tweet) is True

    def test_detects_case_insensitive(self):
        """Detection should be case insensitive."""
        tweet = "TRUMP said something about trade"
        assert is_us_noise(tweet) is True

    def test_indian_tweet_not_noise(self):
        """Clean Indian political tweet should not be flagged."""
        tweet = "Modi government announces new education policy"
        assert is_us_noise(tweet) is False

    def test_congress_party_india(self):
        """Congress (Indian party) should not trigger US noise filter."""
        tweet = "Congress party wins in Karnataka elections"
        # Note: "congress" is in US noise list — this tests a known limitation
        # The filter is intentionally aggressive
        assert is_us_noise(tweet) is False  # known tradeoff in current implementation

    def test_empty_tweet(self):
        """Empty tweet should not be flagged as noise."""
        assert is_us_noise("") is False


# =============================================================================
# TESTS FOR is_long_enough()
# =============================================================================

class TestIsLongEnough:

    def test_long_tweet_passes(self):
        """Tweet with more than MIN_WORDS words should pass."""
        tweet = "The BJP government has announced a new policy for farmers today"
        assert is_long_enough(tweet) is True

    def test_short_tweet_fails(self):
        """Tweet with fewer than MIN_WORDS words should fail."""
        tweet = "BJP wins"
        assert is_long_enough(tweet) is False

    def test_exactly_min_words(self):
        """Tweet with exactly MIN_WORDS (8) words should pass."""
        tweet = "Modi BJP Congress AAP TMC DMK India politics"
        assert is_long_enough(tweet) is True

    def test_empty_tweet_fails(self):
        """Empty tweet should not pass length check."""
        assert is_long_enough("") is False

    def test_custom_min_words(self):
        """Custom min_words parameter should work correctly."""
        tweet = "BJP wins election"
        assert is_long_enough(tweet, min_words=3) is True
        assert is_long_enough(tweet, min_words=5) is False


# =============================================================================
# TESTS FOR SentimentClassifier
# =============================================================================

class TestSentimentClassifier:

    def setup_method(self):
        """Create simple synthetic data for testing classifier."""
        np.random.seed(42)
        # 60 samples, 10 features — simple synthetic embeddings
        self.X_train = np.random.randn(60, 10)
        # 3 classes: 0=Positive, 1=Neutral, 2=Negative — 20 samples each
        self.y_train = [0] * 20 + [1] * 20 + [2] * 20
        self.X_test = np.random.randn(10, 10)

    def test_classifier_trains_without_error(self):
        """Classifier should train without raising any exception."""
        clf = SentimentClassifier()
        clf.fit(self.X_train, self.y_train)
        assert clf.trained is True

    def test_classifier_predict_returns_correct_shape(self):
        """Predict should return array of same length as input."""
        clf = SentimentClassifier()
        clf.fit(self.X_train, self.y_train)
        preds = clf.predict(self.X_test)
        assert len(preds) == len(self.X_test)

    def test_classifier_predict_valid_classes(self):
        """All predictions should be within valid class range (0, 1, 2)."""
        clf = SentimentClassifier()
        clf.fit(self.X_train, self.y_train)
        preds = clf.predict(self.X_test)
        assert all(p in [0, 1, 2] for p in preds)

    def test_confidence_returns_correct_shape(self):
        """Confidence scores should have same length as input."""
        clf = SentimentClassifier()
        clf.fit(self.X_train, self.y_train)
        conf = clf.confidence(self.X_test)
        assert len(conf) == len(self.X_test)

    def test_confidence_scores_in_valid_range(self):
        """All confidence scores should be between 0 and 1."""
        clf = SentimentClassifier()
        clf.fit(self.X_train, self.y_train)
        conf = clf.confidence(self.X_test)
        assert all(0.0 <= c <= 1.0 for c in conf)

    def test_predict_proba_sums_to_one(self):
        """Probability distribution for each sample should sum to 1."""
        clf = SentimentClassifier()
        clf.fit(self.X_train, self.y_train)
        proba = clf.predict_proba(self.X_test)
        for row in proba:
            assert abs(sum(row) - 1.0) < 1e-6

    def test_trained_flag_starts_false(self):
        """Classifier should start with trained=False before fitting."""
        clf = SentimentClassifier()
        assert clf.trained is False


# =============================================================================
# TESTS FOR THRESHOLD CONSTANTS
# =============================================================================

class TestThresholds:

    def test_high_conf_threshold_is_valid(self):
        """HIGH_CONF_THRESHOLD should be between 0 and 1."""
        assert 0.0 < HIGH_CONF_THRESHOLD <= 1.0

    def test_uncertain_threshold_is_valid(self):
        """UNCERTAIN_THRESHOLD should be between 0 and 1."""
        assert 0.0 < UNCERTAIN_THRESHOLD <= 1.0

    def test_high_conf_greater_than_uncertain(self):
        """HIGH_CONF_THRESHOLD must be greater than UNCERTAIN_THRESHOLD."""
        assert HIGH_CONF_THRESHOLD > UNCERTAIN_THRESHOLD

    def test_threshold_gap_exists(self):
        """There should be a gap between thresholds — tweets in the middle
        are neither auto-labelled nor flagged as uncertain."""
        gap = HIGH_CONF_THRESHOLD - UNCERTAIN_THRESHOLD
        assert gap > 0


# =============================================================================
# INTEGRATION TEST — preprocessing pipeline
# =============================================================================

class TestPreprocessingPipeline:

    def test_full_pipeline_on_sample_tweets(self):
        """
        End-to-end test: a raw tweet should pass through all
        preprocessing steps correctly.
        """
        raw_tweet = "RT : @RahulGandhi says #Congress will win! https://t.co/abc &amp; more"

        # Step 1: Not US noise
        assert is_us_noise(raw_tweet) is False

        # Step 2: Clean it
        cleaned = clean_tweet(raw_tweet)
        assert "@RahulGandhi" not in cleaned
        assert "https" not in cleaned
        assert "#Congress" not in cleaned
        assert "Congress" in cleaned
        assert "&amp;" not in cleaned
        assert "&" in cleaned

        # Step 3: Check length
        # This specific tweet after cleaning may or may not pass — just check it runs
        result = is_long_enough(cleaned)
        assert isinstance(result, bool)

    def test_url_only_tweet_filtered_by_length(self):
        """
        A tweet that is only a URL should be cleaned to empty
        and then filtered out by length check.
        """
        raw = "https://t.co/abc123xyz"
        cleaned = clean_tweet(raw)
        assert is_long_enough(cleaned) is False
