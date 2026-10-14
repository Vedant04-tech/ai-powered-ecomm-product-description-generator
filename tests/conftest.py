"""Pytest configuration: isolate tests from the real .env and live API credentials."""
import os
import pytest


@pytest.fixture(autouse=True)
def _no_real_api_key(monkeypatch):
    """Prevent any test from accidentally using a real Gemini API key from the environment.

    Tests that need a key should pass api_key='fake-key-for-test' explicitly.
    This fixture runs for every test automatically.
    """
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
