import pytest

from icl_sentiment.providers.base import ProviderConfig, ProviderRegistry


def test_fake_provider_is_registered(fake_provider_config):
    assert "fake" in ProviderRegistry.available()
    provider = ProviderRegistry.create(fake_provider_config)
    response = provider.classify("some prompt")
    assert response in {"positive", "neutral", "negative"}


def test_unknown_provider_raises():
    config = ProviderConfig(name="does-not-exist", model="x")
    with pytest.raises(KeyError):
        ProviderRegistry.create(config)


def test_provider_label(fake_provider_config):
    provider = ProviderRegistry.create(fake_provider_config)
    assert provider.label == "fake:fake-v1"


def test_resolve_api_key_missing_raises(monkeypatch):
    monkeypatch.delenv("SOME_TEST_KEY", raising=False)
    config = ProviderConfig(name="fake", model="x", api_key_env="SOME_TEST_KEY")
    with pytest.raises(EnvironmentError):
        config.resolve_api_key()


def test_resolve_api_key_present(monkeypatch):
    monkeypatch.setenv("SOME_TEST_KEY", "secret")
    config = ProviderConfig(name="fake", model="x", api_key_env="SOME_TEST_KEY")
    assert config.resolve_api_key() == "secret"
